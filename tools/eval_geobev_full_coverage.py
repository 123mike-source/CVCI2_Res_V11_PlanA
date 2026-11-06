"""Evaluate all six-future-valid validation frames, adding cold-start E2E output.

The original CVCI3 rolling path and its serial CUDA noise bank stay intact.
Frames without three observations use the same checkpoint's GeoBEV/DiT planner
without future-world conditioning. Fallback outputs never enter the rolling
state. Frames without ALL six valid future labels remain excluded from scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import eval_cvci3_protocol as legacy
from geobev_encoder.dataset import GeoBEVEvaluationDataset
from geobev_encoder.model import GeoBEVRollingState


class ObservedRollingState(GeoBEVRollingState):
    def update(self, batch):
        self.current_history = super().update(batch)
        return self.current_history


def configure_worker(worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def token_digest(tokens):
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


def score_records(records, error_field='errors_lidar'):
    """Use only complete trajectories and match the original horizon averages."""
    if not records:
        return dict(frames=0, frames_with_any_label=0, prefix_l2_m=[None]*3,
                    avg_L2_m=None, prefix_valid_frames=[0]*3,
                    endpoint_l2_m=[None]*3, avg_endpoint_L2_m=None,
                    endpoint_valid_frames=[0]*3, point_valid_counts=[0]*6)
    errors = np.asarray([r[error_field] for r in records], dtype=np.float64)
    masks = np.asarray([r['mask'] for r in records], dtype=bool)
    if (errors.shape != masks.shape or errors.shape[1:] != (6,)
            or not masks.all() or not np.isfinite(errors).all()):
        raise AssertionError('invalid per-frame errors/masks')
    prefix, prefix_counts, endpoints, endpoint_counts = [], [], [], []
    for k in (2, 4, 6):
        counts = masks[:, :k].sum(1)
        eligible = counts > 0
        values = (errors[:, :k]*masks[:, :k]).sum(1)
        prefix.append(float((values[eligible]/counts[eligible]).mean()) if eligible.any() else None)
        prefix_counts.append(int(eligible.sum()))
        valid = masks[:, k-1]
        endpoints.append(float(errors[valid, k-1].mean()) if valid.any() else None)
        endpoint_counts.append(int(valid.sum()))
    return dict(frames=len(records), frames_with_any_label=int(masks.any(1).sum()),
                prefix_l2_m=prefix,
                avg_L2_m=float(np.mean(prefix)) if all(v is not None for v in prefix) else None,
                prefix_valid_frames=prefix_counts, endpoint_l2_m=endpoints,
                avg_endpoint_L2_m=float(np.mean(endpoints)) if all(v is not None for v in endpoints) else None,
                endpoint_valid_frames=endpoint_counts, point_valid_counts=masks.sum(0).tolist())


def compare_reference(records, directory, tolerance):
    directory = Path(directory)
    original = {}
    for path in sorted(directory.glob('samples-rank*.jsonl')):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record['token'] in original:
                raise AssertionError('duplicate token in reference')
            original[record['token']] = record
    eligible = [r for r in records if r['ready']]
    maximum = 0.0
    for record in eligible:
        previous = original[record['token']]
        if record['noise_row'] != previous['noise_row'] or record['mask'] != previous['mask']:
            raise AssertionError('reference sample/noise/mask changed')
        difference = float(np.max(np.abs(np.asarray(record['rolling_prediction_lidar']) -
                                         np.asarray(previous['prediction_lidar']))))
        maximum = max(maximum, difference)
    if not eligible or maximum > tolerance:
        raise AssertionError(f'original rolling predictions changed: max difference={maximum} m')
    return dict(status='PASS', compared_frames=len(eligible),
                max_abs_prediction_difference_m=maximum, tolerance_m=tolerance)


def end_to_end_prediction(model, batch, history, noise, args, ddim_indices):
    # Pure scoring readout: no stream.update or stream.set_prediction call here.
    ego, agents, dense, status, _, canbus = model._build_conditions(batch, history, False)
    extra = {}
    if getattr(model, 'plan_a_config', None):
        extra['motion_features'] = torch.zeros(len(ego), 6, 20, device=ego.device, dtype=torch.float32)
    return model.base.planner.sample(len(ego), ego, agents, dense, status,
        canbus=canbus, initial_noise=noise, inference_steps=args.ddim_steps,
        timestep_indices=ddim_indices, future_tokens=None, **extra)


def layer_errors(model, batch, final_prediction):
    """Read all heads from this same t99 pass; never roll a different prior."""
    planner = model.base.planner
    predictions = planner.last_sample_layer_predictions
    if predictions is None or len(predictions) != len(planner.layers):
        raise AssertionError('Missing single-pass layer predictions')
    target = batch['target_lidar'][0, :6, :2].cumsum(0)
    errors = []
    for index, prediction in enumerate(predictions):
        positions = planner.denormalize(planner.clip_normalized(prediction.float())).cumsum(1)
        if index == len(predictions)-1 and not torch.equal(positions, final_prediction):
            raise AssertionError('Layer diagnostic changed the deployed final output')
        lidar = legacy.ego_positions_to_lidar_deltas(positions, batch['lidar_to_ego'])[0].cumsum(0)
        distances = (lidar - target).square().sum(-1).sqrt()
        if not torch.isfinite(distances).all():
            raise FloatingPointError('Nonfinite intermediate layer trajectory')
        errors.append(distances.cpu().tolist())
    return errors


def layer_scores(records):
    return [dict(layer=index + 1, **score_records([
        dict(mask=record['mask'], errors_lidar=record['layer_errors_lidar'][index])
        for record in records])) for index in range(len(records[0]['layer_errors_lidar']) if records else 0)]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-checkpoint', required=True)
    parser.add_argument('--base-checkpoint', default=str(legacy.DEFAULT_BASE))
    parser.add_argument('--world-checkpoint', default=str(legacy.DEFAULT_WORLD))
    parser.add_argument('--world-root', default=str(legacy.DEFAULT_WORLD_ROOT))
    parser.add_argument('--data-root', default=legacy.DEFAULT_DATA_ROOT)
    parser.add_argument('--ann-file', help='default: DATA_ROOT/nuscenes_infos_val.pkl')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-dir', help='original best-checkpoint evaluation, for preservation audit')
    parser.add_argument('--reference-tolerance-m', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--ddim-steps', type=int, default=1)
    parser.add_argument('--layer-diagnostics', action='store_true',
                        help='score all architecture heads from the same t99 call and unchanged final-layer rolling prior')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--max-scenes', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--log-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.layer_diagnostics and args.ddim_steps != 1:
        parser.error('Layer diagnostics require the single-step t99 deployment protocol')
    if (min(args.workers, args.max_scenes, args.max_frames) < 0 or args.prefetch_factor < 1
            or args.log_seconds <= 0 or not 1 <= args.ddim_steps <= 100
            or args.reference_tolerance_m < 0):
        parser.error('invalid evaluation arguments')
    return args


def main():
    args = arguments()
    started = time.monotonic()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '2'))))
    configure_worker(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from world_models.irwm_wrapper import FrozenIRWMFutureBEV
    FrozenIRWMFutureBEV.readout_steps = 5

    output = Path(args.output_dir)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'choose a fresh output directory: {output}')
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    checkpoint = torch.load(args.adapter_checkpoint, map_location='cpu')
    if checkpoint.get('metadata', {}).get('bev_encoder') != 'geobev':
        raise ValueError('This independent evaluator requires the GeoBEV V9 checkpoint')
    annotation = args.ann_file or str(Path(args.data_root)/'nuscenes_infos_val.pkl')
    dataset = GeoBEVEvaluationDataset(args.data_root, annotation)
    scenes, planned, priors, original_complete = legacy.canonical_schedule(
        dataset.infos, args.max_scenes, args.max_frames)
    serial_indices = [i for _, indices in scenes for i in indices]
    warm = {i: k for k, i in enumerate(i for i in serial_indices if i not in planned)}
    assignments, loads = legacy.partition_scenes(scenes, world)
    local_indices = [i for scene in assignments[rank] for i in scenes[scene][1]]
    bank = legacy.cvci3_noise_bank(len(planned), args.seed, device)
    # Separate generator: additional warm frames cannot shift the original bank.
    warm_seed = args.seed + 1000003
    warm_bank = legacy.cvci3_noise_bank(len(warm), warm_seed, device)
    noise_sha = hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest()
    ddim_indices = legacy.ddim_schedule_indices('pretrained', args.ddim_steps)
    model = legacy.build_temporal_model(checkpoint, args.base_checkpoint, args.world_checkpoint,
        args.world_root, enable_world=True, seed=args.seed).to(device).eval()
    model.stream = ObservedRollingState(continuity_bounds=(0.4, 0.6))
    if checkpoint['metadata']['world_sha256'] != legacy.file_sha256(args.world_checkpoint):
        raise RuntimeError('checkpoint/world fingerprint mismatch')
    legacy.load_planning_checkpoint(model, checkpoint)
    model.base.planner.collect_sample_layer_diagnostics = args.layer_diagnostics
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint_sha = legacy.file_sha256(args.adapter_checkpoint)
    if args.reference_dir:
        reference = json.loads((Path(args.reference_dir)/'config.json').read_text())
        if (reference['adapter_sha256'] != checkpoint_sha or reference['seed'] != args.seed
                or reference['ddim_indices'] != ddim_indices):
            raise AssertionError('reference checkpoint/seed/DDIM do not match')
        if not (args.max_scenes or args.max_frames) and reference['noise_sha256'] != noise_sha:
            raise AssertionError('original serial noise bank changed')
    metadata = dict(evaluation_protocol='geobev_full_coverage_scoring_fallback_v1',
        split='val' if args.ann_file is None else 'custom_annotated', annotation_file=annotation,
        annotation_sha256=legacy.file_sha256(annotation), adapter_checkpoint=args.adapter_checkpoint,
        adapter_sha256=checkpoint_sha, adapter_epoch=checkpoint.get('epoch'),
        architecture_version=model.architecture_version, seed=args.seed,
        ddim_indices=ddim_indices, ddim_steps=args.ddim_steps, noise_sha256=noise_sha,
        warmup_noise_seed=warm_seed, warmup_noise_sha256=hashlib.sha256(warm_bank.cpu().numpy().tobytes()).hexdigest(),
        selection_rule='score only all_six_future_labels_valid; E2E iff history_length<3',
        evaluation_scope='all annotated validation frames; exclude incomplete future labels',
        fallback='same_checkpoint_geobev_dit_without_future_world_condition',
        fallback_updates_rolling_state=False, branch_selection_uses_gt=False,
        cold_start_frames=3, continuity_seconds=[0.4, 0.6], world_readout_steps=5,
        metric_frame='lidar', metric_filter='all_six_future_points_valid',
        missing_labels='discard_entire_trajectory_if_any_future_point_invalid',
        main_metric='prefix_mean_at_1_2_3s_then_mean_over_horizons_same_as_original',
        expected_seen_frames=len(serial_indices), expected_original_planning_frames=len(planned),
        expected_original_complete_samples=len(original_complete), world_size=world,
        partial=bool(args.max_scenes or args.max_frames), reference_dir=args.reference_dir,
        layer_diagnostics=args.layer_diagnostics,
        script_sha256=legacy.file_sha256(__file__), original_evaluator_sha256=legacy.file_sha256(legacy.__file__),
        backend=dict(cudnn_benchmark=False, cudnn_deterministic=True, tf32=True))
    if rank == 0:
        legacy.atomic_json(output/'config.json', dict(metadata, args=vars(args)))
        print(json.dumps(metadata), flush=True)
    loader_args = dict(batch_sampler=[[i] for i in local_indices], num_workers=args.workers,
        collate_fn=legacy.temporal_collate, pin_memory=True, persistent_workers=args.workers > 0,
        multiprocessing_context='spawn' if args.workers else None,
        generator=torch.Generator().manual_seed(args.seed + rank + 900000))
    if args.workers:
        loader_args.update(prefetch_factor=args.prefetch_factor, worker_init_fn=configure_worker)
    loader = DataLoader(dataset, **loader_args)
    state = dict(rank=rank, frames=0, total_frames=len(local_indices),
                 complete_samples=0, planning_frames=0, fallback_frames=0,
                 fps=0, done=False, phase='evaluating')
    def snapshot(done=False):
        state.update(done=done, fps=state['frames']/max(time.monotonic()-started, 1e-9),
                     updated_at=time.time())
        legacy.atomic_json(output/f'progress-rank{rank}.json', state)
    snapshot()
    last_log = time.monotonic()
    with (output/f'samples-rank{rank}.jsonl').open('w') as stream, torch.no_grad():
        for number, cpu_batch in enumerate(loader, 1):
            index = int(cpu_batch['data_index'][0])
            batch = legacy.to_device(cpu_batch, device)
            row = planned.get(index)
            noise = bank[row:row+1] if row is not None else warm_bank[warm[index]:warm[index]+1]
            if args.layer_diagnostics:
                model.base.planner.last_sample_layer_predictions = None
            result = model(batch, compute_loss=False, disable_future=False,
                initial_noise=noise if row is not None else None, ddim_steps=args.ddim_steps,
                timestep_indices=ddim_indices, skip_warmup=True)
            ready, prior_valid = bool(result['ready'][0]), bool(result['prior_valid'][0])
            if ready != (row is not None) or prior_valid != priors.get(index, False):
                raise AssertionError(f'original cold-start schedule changed at index {index}')
            mask = batch['mask'][0, :6].bool().cpu().tolist()
            complete = all(mask)
            fallback = not ready and complete
            prediction_ego = None
            if fallback:
                prediction_ego = end_to_end_prediction(model, batch,
                    model.stream.current_history, noise, args, ddim_indices)
            elif complete:
                prediction_ego = result['prediction']
            record = dict(token=batch['meta'][0]['token'], scene=batch['meta'][0]['scene_token'],
                data_index=index, noise_row=row, warmup_noise_row=warm.get(index),
                ready=ready, prior_valid=prior_valid, complete=complete, mask=mask,
                scored=complete, source=('e2e' if fallback else 'rolling') if complete else 'discarded',
                exclusion_reason=None if complete else 'incomplete_future',
                prediction_lidar=None, prediction_ego=None, errors_lidar=None, errors_ego=None)
            if complete:
                prediction_lidar = legacy.ego_positions_to_lidar_deltas(
                    prediction_ego, batch['lidar_to_ego'])[0].cumsum(0)
                target_lidar = batch['target_lidar'][0, :6, :2].cumsum(0)
                target_ego = batch['trajectory_delta'][0, :6, :2].cumsum(0)
                # Match the original evaluator's arithmetic as well as its formula.
                error_lidar = (prediction_lidar-target_lidar).square().sum(-1).sqrt()
                error_ego = (prediction_ego[0]-target_ego).square().sum(-1).sqrt()
                if not torch.isfinite(error_lidar).all() or not torch.isfinite(error_ego).all():
                    raise FloatingPointError(f'nonfinite output at {index}')
                record.update(prediction_lidar=prediction_lidar.cpu().tolist(),
                    prediction_ego=prediction_ego[0].cpu().tolist(),
                    errors_lidar=error_lidar.cpu().tolist(), errors_ego=error_ego.cpu().tolist())
                if args.layer_diagnostics:
                    record['layer_errors_lidar'] = layer_errors(model, batch, prediction_ego)
            if ready:
                original_lidar = legacy.ego_positions_to_lidar_deltas(
                    result['prediction'], batch['lidar_to_ego'])[0].cumsum(0)
                record['rolling_prediction_lidar'] = original_lidar.cpu().tolist()
            stream.write(json.dumps(record)+'\n')
            state.update(frames=number, planning_frames=state['planning_frames']+int(ready),
                complete_samples=state['complete_samples']+int(ready and complete),
                fallback_frames=state['fallback_frames']+int(fallback))
            if time.monotonic()-last_log >= args.log_seconds:
                stream.flush()
                snapshot()
                if rank == 0:
                    legacy.global_progress(output, world, loads, started)
                last_log = time.monotonic()
    torch.cuda.synchronize(device)
    snapshot(done=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        records = []
        for part in range(world):
            records.extend(json.loads(line) for line in (output/f'samples-rank{part}.jsonl').read_text().splitlines())
        by_token = {r['token']: r for r in records}
        expected = [str(dataset.infos[i]['token']) for i in serial_indices]
        if len(by_token) != len(records) or set(by_token) != set(expected):
            raise AssertionError('duplicate or missing full-coverage predictions')
        records = [by_token[token] for token in expected]
        scored = [r for r in records if r['scored']]
        retained = [r for r in records if r['source'] == 'rolling']
        expected_scored = [str(dataset.infos[i]['token']) for i in serial_indices
                           if bool(np.asarray(dataset.infos[i]['gt_ego_fut_masks'])[:6].all())]
        if [r['token'] for r in scored] != expected_scored:
            raise AssertionError('full-valid evaluation omitted or included unexpected frames')
        if [r['token'] for r in retained] != original_complete:
            raise AssertionError('original complete-sample selection changed')
        audit = compare_reference(records, args.reference_dir, args.reference_tolerance_m) if args.reference_dir else None
        metrics = dict(metadata, seen_frames=len(records), scored_frames=len(scored),
            complete_samples=len(scored), rolling_frames=len(retained),
            fallback_frames=len(scored)-len(retained),
            original_planning_frames=sum(r['ready'] for r in records),
            discarded_incomplete_future_frames=len(records)-len(scored),
            full_six_valid=score_records(scored),
            full_six_valid_ego=score_records(scored, 'errors_ego'),
            retained_original_subset=score_records(retained),
            fallback_subset=score_records([r for r in records if r['source'] == 'e2e']),
            reference_preservation_audit=audit, sample_coverage_check='PASS',
            elapsed_seconds=time.monotonic()-started)
        metrics['avg_L2_m'] = metrics['full_six_valid']['avg_L2_m']
        metrics['prefix_l2'] = metrics['full_six_valid']['prefix_l2_m']
        metrics.update({f'L2@{k+1}s_m': value for k, value in enumerate(metrics['prefix_l2'])})
        if args.layer_diagnostics:
            diagnostic = dict(
                protocol='t99_all_heads_same_noise_context_and_final_layer_rolling_prior_v2',
                interpretation='head readout diagnostic; not independently rolled planners',
                validation_token_sha256=token_digest(expected_scored),
                checkpoint_sha256=checkpoint_sha, noise_sha256=noise_sha,
                full_six_valid=layer_scores(scored),
                retained_original_subset=layer_scores(retained),
                fallback_subset=layer_scores([r for r in scored if r['source'] == 'e2e']))
            for key in ('full_six_valid', 'retained_original_subset', 'fallback_subset'):
                if diagnostic[key][-1] != dict(layer=len(model.base.planner.layers), **metrics[key]):
                    raise AssertionError('Final-layer L2 diagnostic differs from the primary metric')
            diagnostic['final_layer_metric_check'] = 'PASS'
            legacy.atomic_json(output/'layer_diagnostics.json', diagnostic)
            metrics['layer_diagnostics_file'] = str(output/'layer_diagnostics.json')
        legacy.atomic_json(output/'metrics.json', metrics)
        legacy.global_progress(output, world, loads, started, final=True)
        print('FULL_COVERAGE_RESULT '+json.dumps(metrics), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
