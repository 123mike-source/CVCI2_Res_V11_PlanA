"""Paired cold-start evaluation: sample-token noise, prefix AND endpoint metrics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import DEFAULT_DATA_ROOT
from temporal_dataset import TemporalNuScenesDataset, temporal_collate, to_device
from planning_runtime import DEFAULT_BASE, DEFAULT_WORLD, DEFAULT_WORLD_ROOT, file_sha256
from planning_runtime import load_planning_checkpoint, build_temporal_model, planning_model_metadata
from world_models.rolling_state import TRAJECTORY_TRANSFORM_VERSION


def main():
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
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-scenes', type=int, default=0, help='global scene limit; 0=all')
    parser.add_argument('--max-frames', type=int, default=0, help='per-rank smoke limit; 0=all')
    parser.add_argument('--log-every', type=int, default=20)
    args = parser.parse_args()
    if args.mode != 'baseline' and not args.adapter_checkpoint:
        parser.error('future/disabled requires --adapter-checkpoint')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group('nccl')
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.set_num_threads(min(4, max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))))
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.adapter_checkpoint, map_location='cpu') if args.adapter_checkpoint and args.mode != 'baseline' else None
    model = build_temporal_model(checkpoint, args.base_checkpoint, args.world_checkpoint, args.world_root,
                          enable_world=args.mode == 'future', seed=args.seed).to(device).eval()
    if checkpoint is not None:
        if args.mode == 'future' and checkpoint['metadata']['world_sha256'] != file_sha256(args.world_checkpoint):
            raise RuntimeError('adapter/world fingerprint mismatch')
        if args.mode != 'baseline':
            load_planning_checkpoint(model, checkpoint)
    dataset = TemporalNuScenesDataset(args.data_root, str(Path(args.data_root) / 'nuscenes_infos_val.pkl'))
    scenes = list(dataset.scenes.values())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    indices = [i for scene in scenes[rank::world] for i in scene]
    if args.max_frames:
        indices = indices[:args.max_frames]
    loader = DataLoader(dataset, batch_sampler=[[i] for i in indices], num_workers=args.workers,
                        collate_fn=temporal_collate, pin_memory=True,
                        multiprocessing_context='spawn' if args.workers else None,
                        persistent_workers=args.workers > 0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # prefix sums(3), complete count, endpoint sums(3), endpoint counts(3), seen, ready
    totals = torch.zeros(12, device=device, dtype=torch.float64)
    stage = {}
    started = time.perf_counter()
    data_end = started
    load_seconds = 0.0
    with open(output_dir / f'samples-rank{rank}.jsonl', 'w') as stream, torch.no_grad():
        for number, batch in enumerate(loader, 1):
            load_seconds += time.perf_counter() - data_end
            begin = time.perf_counter()
            batch = to_device(batch, device)
            torch.cuda.synchronize()
            stage['transfer'] = stage.get('transfer', 0) + time.perf_counter() - begin
            result = model(batch, compute_loss=False, disable_future=args.mode != 'future', profile=True)
            for key, value in result['timings'].items():
                stage[key] = stage.get(key, 0) + value
            totals[10] += 1
            if bool(result['ready'][0]):
                error = (result['prediction'][0] - batch['trajectory_delta'][0].cumsum(0)).norm(dim=-1).double()
                mask = batch['mask'][0].bool()
                totals[11] += 1
                for slot, point in enumerate((1, 3, 5)):
                    totals[4 + slot] += error[point] * mask[point]
                    totals[7 + slot] += mask[point]
                if bool(mask.all()):
                    totals[:3] += torch.stack([error[:k].mean() for k in (2, 4, 6)])
                    totals[3] += 1
                stream.write(json.dumps({'token': batch['meta'][0]['token'],
                                         'scene': batch['meta'][0]['scene_token'],
                                         'errors': error.cpu().tolist(), 'mask': mask.cpu().tolist(),
                                         'prior_valid': bool(result['prior_valid'][0]),
                                         'prediction_ego': result['prediction'][0].cpu().tolist()}) + '\n')
            if rank == 0 and number % args.log_every == 0:
                print(f'frames={number}/{len(indices)} complete={int(totals[3])} seconds={time.perf_counter()-started:.1f}', flush=True)
            data_end = time.perf_counter()
    if distributed:
        dist.all_reduce(totals)
    elapsed = time.perf_counter() - started
    metrics = {'mode': args.mode, 'seed': args.seed, 'ddim_indices': [99, 74, 50, 25, 0],
               'trajectory_transform': TRAJECTORY_TRANSFORM_VERSION,
               'checkpoint_format': checkpoint.get('format') if checkpoint else None,
               'cold_start_frames': 3, 'complete_samples': int(totals[3]),
               'seen_frames': int(totals[10]), 'planning_frames': int(totals[11]),
               'prefix_l2': (totals[:3] / totals[3].clamp_min(1)).cpu().tolist(),
               'endpoint_l2': (totals[4:7] / totals[7:10].clamp_min(1)).cpu().tolist(),
               'endpoint_counts': totals[7:10].cpu().tolist(),
               'rank': rank, 'rank_elapsed_seconds': elapsed, 'rank_data_wait_seconds': load_seconds,
               'rank_stage_seconds': stage, 'rank_max_memory_mib': torch.cuda.max_memory_allocated()/2**20,
               'partial': bool(args.max_scenes or args.max_frames),
               'base_checkpoint': args.base_checkpoint, 'adapter_checkpoint': args.adapter_checkpoint}
    metrics.update(planning_model_metadata(model))
    metrics['prefix_avg'] = sum(metrics['prefix_l2']) / 3 if metrics['complete_samples'] else None
    (output_dir / f'metrics-rank{rank}.json').write_text(json.dumps(metrics, indent=2))
    if rank == 0:
        (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
