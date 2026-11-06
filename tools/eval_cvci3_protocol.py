"""CVCI3's serial cold-start protocol, partitioned over complete scenes.

Main scores reproduce tools/eval_cvci3_temporal.py in CVCI3: 3-frame
cold start, inclusive 0.4..0.6 second continuity, six-valid-point filtering,
LiDAR-frame error, seed=2026, and (by default) DDIM 80/60/40/20/0.
Use --ddim-schedule pretrained for the base planner's original linspace
schedule: 99/74/50/25/0 for five steps. The other protocol settings stay fixed.
CUDA planner noise is generated in the reference's serial scene order before
partitioning, so changing GPU count never changes a sample's random draw.
"""
import argparse
from collections import OrderedDict
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import DEFAULT_DATA_ROOT
from temporal_dataset import TemporalNuScenesDataset, temporal_collate, to_device
from planning_runtime import DEFAULT_BASE, DEFAULT_WORLD, DEFAULT_WORLD_ROOT, file_sha256
from planning_runtime import load_planning_checkpoint, build_temporal_model, planning_model_metadata
from world_models.rolling_state import (RollingState, ego_positions_to_lidar_deltas,
                                        TRAJECTORY_TRANSFORM_VERSION)

ROOT = Path(__file__).resolve().parents[1]


class EvaluationDataset(TemporalNuScenesDataset):
    def __getitem__(self, index):
        output = super().__getitem__(index)
        output['target_lidar'] = torch.as_tensor(
            np.asarray(self.infos[index]['gt_ego_fut_trajs'], dtype=np.float32))
        return output


def canonical_schedule(infos, max_scenes=0, max_frames=0):
    # NuScenes3DDetTrackDataset.load_annotations sorts globally by timestamp;
    # CVCI3 then groups scenes in that order, preserving each scene's order.
    scenes = OrderedDict()
    for index in sorted(range(len(infos)), key=lambda i: infos[i]['timestamp']):
        scenes.setdefault(str(infos[index]['scene_token']), []).append(index)
    selected = list(scenes.items())[:max_scenes or None]
    result, planned, priors, complete = [], {}, {}, []
    remaining = max_frames or sum(len(indices) for _, indices in selected)
    for scene, original in selected:
        indices = original[:remaining]
        remaining -= len(indices)
        if not indices:
            break
        history, previous_time, previous_plan = 0, None, False
        for index in indices:
            info = infos[index]
            timestamp = float(info['timestamp']) / 1e6
            continuous = previous_time is not None and 0.4 <= timestamp - previous_time <= 0.6
            if not continuous:
                history, previous_plan = 0, False
            history += 1
            if history >= 3:
                planned[index] = len(planned)
                priors[index] = previous_plan
                previous_plan = True
                if bool(np.asarray(info['gt_ego_fut_masks'])[:6].all()):
                    complete.append(str(info['token']))
            previous_time = timestamp
        result.append((scene, indices))
    return result, planned, priors, complete


def partition_scenes(scenes, world):
    # Greedy whole-scene assignment balances work without splitting histories.
    assignment, loads = [[] for _ in range(world)], [0] * world
    for order, (_, indices) in sorted(enumerate(scenes), key=lambda x: (-len(x[1][1]), x[0])):
        rank = min(range(world), key=lambda r: (loads[r], r))
        assignment[rank].append(order)
        loads[rank] += len(indices)
    return [sorted(values) for values in assignment], loads


def cvci3_noise_bank(count, seed, device):
    # Keep the SAME call shape and call count as serial CVCI3 torch.randn.
    # A single large randn has different Philox offsets and is not equivalent.
    generator = torch.Generator(device=device).manual_seed(seed)
    bank = torch.empty(count, 6, 2, device=device)
    for row in range(count):
        torch.randn(1, 6, 2, generator=generator, device=device, out=bank[row:row + 1])
    return bank


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def ddim_schedule_indices(schedule, steps, train_timesteps=100):
    if not 1 <= steps <= train_timesteps:
        raise ValueError('DDIM step count must be in [1, train_timesteps]')
    if schedule == 'pretrained':
        return torch.linspace(train_timesteps - 1, 0, steps).round().long().tolist()
    if schedule == 'cvci3':
        return [i * max(train_timesteps // steps, 1) for i in range(steps - 1, -1, -1)]
    raise ValueError(f'unknown DDIM schedule: {schedule}')


def duration(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def read_progress_snapshots(output, world):
    values = []
    for rank in range(world):
        try:
            values.append(json.loads((output / f'progress-rank{rank}.json').read_text()))
        except OSError as error:
            # Shared FS metadata can transiently report ENOENT/ESTALE even
            # across atomic replace. Progress is advisory; retry next poll.
            if error.errno not in (errno.ENOENT, errno.ESTALE, errno.EAGAIN):
                raise
        except json.JSONDecodeError:
            # Retry a temporarily inconsistent progress snapshot, never metrics.
            continue
    return values


def global_progress(output, world, loads, started, final=False):
    values = read_progress_snapshots(output, world)
    seen = sum(v['frames'] for v in values)
    total = sum(loads)
    elapsed = time.monotonic() - started
    # Per-rank remaining time handles ranks finishing at different moments.
    eta = max([(loads[v['rank']] - v['frames']) / max(v['fps'], 1e-9)
               for v in values if not v['done']] or [0]) if len(values) == world else None
    if any(not v['done'] and v['fps'] == 0 for v in values):
        eta = None
    complete = sum(v['complete_samples'] for v in values)
    plans = sum(v['planning_frames'] for v in values)
    ranks_done = sum(v['done'] for v in values)
    stale = max([time.time() - v['updated_at'] for v in values] or [0])
    print(f'progress frames={seen}/{total} ({100 * seen / max(total, 1):.1f}%) '
          f'planning={plans} complete_samples={complete} '
          f'fps={seen / max(elapsed, 1e-9):.2f} elapsed={duration(elapsed)} '
          f'eta={duration(eta)} ranks_done={ranks_done}/{world} '
          f'snapshot_age_s={stale:.1f}' + (' FINAL' if final else ''), flush=True)
    return len(values) == world and all(v['done'] for v in values)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'disabled', 'future'], default='future')
    parser.add_argument('--adapter-checkpoint')
    parser.add_argument('--base-checkpoint', default=str(DEFAULT_BASE))
    parser.add_argument('--world-checkpoint', default=str(DEFAULT_WORLD))
    parser.add_argument('--world-root', default=str(DEFAULT_WORLD_ROOT))
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='nuScenes data root (default: %(default)s)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--ddim-steps', type=int, default=1)
    parser.add_argument('--ddim-schedule', choices=['cvci3', 'pretrained'], default='pretrained')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--log-seconds', type=float, default=10.0)
    parser.add_argument('--max-scenes', type=int, default=0, help='global prefix scene limit, smoke only')
    parser.add_argument('--max-frames', type=int, default=0, help='global serial prefix frame limit, smoke only')
    parser.add_argument('--profile', action='store_true', help='synchronize stages for timing; slower')
    args = parser.parse_args()
    if args.mode != 'baseline' and not args.adapter_checkpoint:
        parser.error('future/disabled requires a current --adapter-checkpoint')
    if (not 1 <= args.ddim_steps <= 100 or min(args.workers, args.max_scenes, args.max_frames) < 0 or
            args.prefetch_factor < 1 or args.log_every < 1 or args.log_seconds <= 0):
        parser.error('invalid evaluation arguments')
    return args


def main():
    args = arguments()
    started = time.monotonic()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('nccl')
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))
    output = Path(args.output_dir)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'choose a fresh --output-dir: {output}')
        output.mkdir(parents=True, exist_ok=True)
        print(f'initializing protocol=cvci3_cold_start_3_frames GPUs={world} '
              f'workers_per_gpu={args.workers} mode={args.mode} seed={args.seed}', flush=True)
    if distributed:
        dist.barrier()
    checkpoint = torch.load(args.adapter_checkpoint, map_location='cpu') if args.mode != 'baseline' else None
    dataset_type = EvaluationDataset
    if checkpoint is not None and checkpoint.get('metadata',{}).get('bev_encoder') == 'geobev':
        from geobev_encoder.dataset import GeoBEVEvaluationDataset
        dataset_type = GeoBEVEvaluationDataset
    dataset = dataset_type(args.data_root, str(Path(args.data_root) / 'nuscenes_infos_val.pkl'))
    scenes, planned, priors, complete_tokens = canonical_schedule(dataset.infos, args.max_scenes, args.max_frames)
    assignments, loads = partition_scenes(scenes, world)
    local_scenes = [scenes[i] for i in assignments[rank]]
    indices = [index for _, values in local_scenes for index in values]
    state = {'rank': rank, 'frames': 0, 'total_frames': len(indices), 'planning_frames': 0,
             'complete_samples': 0, 'fps': 0.0, 'done': False, 'phase': 'loading_model'}
    def snapshot(done=False):
        state.update(done=done, updated_at=time.time(),
                     fps=state['frames'] / max(time.monotonic() - started, 1e-9))
        atomic_json(output / f'progress-rank{rank}.json', state)
    snapshot()
    bank = cvci3_noise_bank(len(planned), args.seed, device)
    noise_sha = hashlib.sha256(bank.cpu().contiguous().numpy().tobytes()).hexdigest()
    ddim_indices = ddim_schedule_indices(args.ddim_schedule, args.ddim_steps)
    model = build_temporal_model(checkpoint, args.base_checkpoint, args.world_checkpoint, args.world_root,
                          enable_world=args.mode == 'future', seed=args.seed).to(device).eval()
    stream_type = RollingState
    if hasattr(model, 'geobev_config'):
        from geobev_encoder.model import GeoBEVRollingState
        stream_type = GeoBEVRollingState
    model.stream = stream_type(continuity_bounds=(0.4, 0.6))
    if args.mode != 'baseline':
        if args.mode == 'future' and checkpoint['metadata']['world_sha256'] != file_sha256(args.world_checkpoint):
            raise RuntimeError('adapter/world fingerprint mismatch')
        load_planning_checkpoint(model, checkpoint)
    metadata = {'evaluation_protocol': 'cvci3_cold_start_3_frames', 'seed': args.seed,
                'ddim_schedule': args.ddim_schedule,
                'ddim_steps': args.ddim_steps, 'ddim_indices': ddim_indices, 'eta': 0,
                'cold_start_frames': 3, 'first_plan_future_mask': 0, 'rolling_future_mask': 1,
                'continuity_seconds': [0.4, 0.6], 'metric_frame': 'lidar',
                'metric_filter': 'all_six_future_points_valid',
                'validation_planning_metric': 'ego_endpoint_1_2_3s_individual_masks_after_cold_start',
                'noise_protocol': 'serial_cvci3_cuda_randn_1x6x2_before_scene_sharding',
                'noise_sha256': noise_sha, 'scene_order': 'first appearance after global timestamp sort',
                'world_size': world, 'frames_per_rank': loads,
                'expected_seen_frames': sum(loads), 'expected_planning_frames': len(planned),
                'expected_complete_samples': len(complete_tokens),
                'expected_complete_token_sha256': hashlib.sha256('\n'.join(complete_tokens).encode()).hexdigest(),
                'mode': args.mode, 'partial': bool(args.max_scenes or args.max_frames),
                'adapter_checkpoint': args.adapter_checkpoint if args.mode != 'baseline' else None,
                'adapter_epoch': checkpoint.get('epoch') if checkpoint else None,
                'checkpoint_format': checkpoint.get('format') if checkpoint else None,
                'architecture_version': checkpoint['metadata']['architecture_version'] if checkpoint else 0,
                'trajectory_transform': TRAJECTORY_TRANSFORM_VERSION,
                'fusion_config': checkpoint['metadata'].get('fusion_config') if checkpoint else None,
                'training_scope': checkpoint['metadata'].get('training_scope', 'future_adapter_only') if checkpoint else None,
                'adapter_sha256': file_sha256(args.adapter_checkpoint) if checkpoint else None,
                **planning_model_metadata(model),
                'base_checkpoint': args.base_checkpoint, 'world_checkpoint': args.world_checkpoint,
                'workers_per_gpu': args.workers, 'batch_size_per_gpu': 1,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
    if rank == 0:
        atomic_json(output / 'config.json', {**metadata, 'args': vars(args)})
        print(json.dumps(metadata), flush=True)
        print(f'evaluation_protocol=cold_start_3_frames seed={args.seed} ddim_steps={args.ddim_steps} '
              f'ddim_indices={ddim_indices} first_plan_future_mask=0 rolling_future_mask=1 '
              f'expected_complete_samples={len(complete_tokens)}', flush=True)
    loader_options = dict(batch_sampler=[[i] for i in indices], num_workers=args.workers,
                          collate_fn=temporal_collate, pin_memory=True,
                          persistent_workers=args.workers > 0,
                          multiprocessing_context='spawn' if args.workers else None,
                          generator=torch.Generator().manual_seed(args.seed + rank + 900000))
    if args.workers:
        loader_options['prefetch_factor'] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_options)
    # Six complete-sample LiDAR errors + counts, then three individually
    # masked ego endpoint sums/counts (the pretrained validation definition).
    totals = np.zeros(16, dtype=np.float64)
    stage, data_wait = {}, 0.0
    previous_end = time.monotonic()
    last_snapshot = 0.0
    state['phase'] = 'evaluating'
    with open(output / f'samples-rank{rank}.jsonl', 'w') as stream, torch.no_grad():
        for number, cpu_batch in enumerate(loader, 1):
            data_wait += time.monotonic() - previous_end
            index = int(cpu_batch['data_index'][0])
            batch = to_device(cpu_batch, device)
            row = planned.get(index)
            result = model(batch, compute_loss=False, disable_future=args.mode != 'future',
                           profile=args.profile, initial_noise=None if row is None else bank[row:row + 1],
                           ddim_steps=args.ddim_steps, timestep_indices=ddim_indices, skip_warmup=True)
            ready, valid = bool(result['ready'][0]), bool(result['prior_valid'][0])
            if ready != (row is not None) or valid != priors.get(index, False):
                raise AssertionError(f'cold-start state diverged from serial CVCI3 at index {index}')
            totals[7] += 1
            if ready:
                # Base model emits ego XY cumulative points; score in CVCI3's
                # LiDAR XY frame against the original LiDAR GT deltas.
                prediction = ego_positions_to_lidar_deltas(result['prediction'], batch['lidar_to_ego'])[0].cumsum(0)
                ground_truth = batch['target_lidar'][0, :6, :2].cumsum(0)
                error = (prediction - ground_truth).square().sum(-1).sqrt().cpu().numpy()
                ego_target = batch['trajectory_delta'][0, :6, :2].cumsum(0)
                ego_error = (result['prediction'][0] - ego_target).square().sum(-1).sqrt().cpu().numpy()
                point_mask = batch['mask'][0, :6].bool().cpu().numpy()
                if not np.isfinite(error).all() or not np.isfinite(ego_error).all():
                    raise FloatingPointError(f'non-finite planning error at index {index}')
                for slot, point in enumerate((1, 3, 5)):
                    if point_mask[point]:
                        totals[10 + slot] += ego_error[point]
                        totals[13 + slot] += 1
                complete = bool(batch['mask'][0, :6].all())
                if complete:
                    totals[:6] += error
                    totals[6] += 1
                totals[8] += 1
                totals[9] += valid
                stream.write(json.dumps({'token': batch['meta'][0]['token'],
                    'scene': batch['meta'][0]['scene_token'], 'data_index': index,
                    'noise_row': row, 'complete': complete, 'prior_valid': valid,
                    'mask': batch['mask'][0].bool().cpu().tolist(),
                    'prediction_lidar': prediction.cpu().tolist(),
                    'prediction_ego': result['prediction'][0].cpu().tolist(),
                    'errors_ego': ego_error.tolist(),
                    'errors': error.tolist()}) + '\n')
            for key, value in result['timings'].items():
                stage[key] = stage.get(key, 0.0) + value
            state.update(frames=number, planning_frames=int(totals[8]), complete_samples=int(totals[6]))
            now = time.monotonic()
            if number % args.log_every == 0 or now - last_snapshot >= args.log_seconds:
                snapshot()
                if rank == 0:
                    global_progress(output, world, loads, started)
                last_snapshot = now
            previous_end = time.monotonic()
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    state['phase'] = 'finished'
    snapshot(done=True)
    rank_metrics = {'rank': rank, 'local_totals': totals.tolist(), 'elapsed_seconds': elapsed,
                    'data_wait_seconds': data_wait, 'max_memory_mib': torch.cuda.max_memory_allocated()/2**20,
                    'stage_seconds': stage if args.profile else None,
                    'noise_sha256': noise_sha}
    atomic_json(output / f'metrics-rank{rank}.json', rank_metrics)
    if rank == 0:
        while not global_progress(output, world, loads, started):
            time.sleep(args.log_seconds)
    reduced = torch.as_tensor(totals, device=device)
    if distributed:
        dist.all_reduce(reduced)
    totals = reduced.cpu().numpy()
    if rank == 0:
        if (int(totals[6]) != len(complete_tokens) or int(totals[7]) != sum(loads) or
                int(totals[8]) != len(planned)):
            raise AssertionError('global sample counts differ from the serial CVCI3 schedule')
        records = []
        for r in range(world):
            records.extend(json.loads(line) for line in
                           (output / f'samples-rank{r}.jsonl').read_text().splitlines())
        actual_tokens = [r['token'] for r in records if r['complete']]
        if len(actual_tokens) != len(set(actual_tokens)) or set(actual_tokens) != set(complete_tokens):
            raise AssertionError('distributed evaluation duplicated or omitted complete samples')
        points = totals[:6] / max(totals[6], 1)
        prefix = [float(points[:k].mean()) for k in (2, 4, 6)]
        planning_endpoints = [float(points[k]) for k in (1, 3, 5)]
        if (totals[13:16] <= 0).any():
            raise AssertionError('validation planning metric has an empty horizon')
        validation_planning = (totals[10:13] / totals[13:16]).tolist()
        metrics = {**metadata, 'endpoint_L2_m': points.tolist(), 'prefix_l2': prefix,
                   'planning_endpoint_L2_m': planning_endpoints,
                   'avg_endpoint_L2_m': float(np.mean(planning_endpoints)),
                   'validation_planning_L2_m': validation_planning,
                   'validation_planning_avg_L2_m': float(np.mean(validation_planning)),
                   'validation_planning_valid_counts': totals[13:16].astype(int).tolist(),
                   'L2@1s_m': prefix[0], 'L2@2s_m': prefix[1], 'L2@3s_m': prefix[2],
                   'avg_L2_m': float(np.mean(prefix)), 'complete_samples': int(totals[6]),
                   'seen_frames': int(totals[7]), 'planning_frames': int(totals[8]),
                   'prior_valid_frames': int(totals[9]),
                   'elapsed_seconds': time.monotonic() - started,
                   'sample_coverage_check': 'PASS'}
        atomic_json(output / 'metrics.json', metrics)
        global_progress(output, world, loads, started, final=True)
        print('endpoint_L2_m=' + ','.join(f'{value:.6f}' for value in points), flush=True)
        print('validation_planning_L2_m=' + ','.join(f'{v:.6f}' for v in validation_planning) +
              f' validation_planning_avg_L2_m={np.mean(validation_planning):.6f}', flush=True)
        print(f'L2@1s_m={prefix[0]:.6f} L2@2s_m={prefix[1]:.6f} '
              f'L2@3s_m={prefix[2]:.6f} avg_L2_m={np.mean(prefix):.6f} '
              f'complete_samples={int(totals[6])}', flush=True)
        print(json.dumps(metrics), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
