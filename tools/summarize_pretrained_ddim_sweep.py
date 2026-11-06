"""Rank complete validation runs; never select by training diffusion loss."""
import argparse
import csv
import json
import math
import os
from pathlib import Path


PRIMARY = 'validation_planning_avg_L2_m'
PROTOCOL_KEYS = (
    'evaluation_protocol', 'seed', 'ddim_schedule', 'ddim_steps', 'ddim_indices',
    'eta', 'cold_start_frames', 'first_plan_future_mask', 'rolling_future_mask',
    'continuity_seconds', 'metric_frame', 'metric_filter', 'validation_planning_metric',
    'noise_protocol', 'noise_sha256', 'expected_complete_token_sha256',
    'expected_seen_frames', 'expected_planning_frames', 'expected_complete_samples',
    'world_size', 'frames_per_rank', 'base_checkpoint', 'world_checkpoint',
    'workers_per_gpu', 'batch_size_per_gpu', 'cudnn_benchmark',
    'matmul_allow_tf32', 'cudnn_allow_tf32', 'validation_planning_valid_counts',
)


def summarize(root, require_complete=False):
    runs = []
    for name in ['baseline'] + [f'epoch_{epoch:02d}' for epoch in range(1, 11)]:
        path = root / name / 'metrics.json'
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        if (value['partial'] or value['sample_coverage_check'] != 'PASS' or
                value['ddim_schedule'] != 'pretrained' or
                value['ddim_indices'] != [99, 74, 50, 25, 0] or
                value['seen_frames'] != value['expected_seen_frames'] or
                value['planning_frames'] != value['expected_planning_frames'] or
                value['complete_samples'] != value['expected_complete_samples']):
            raise ValueError(f'incomplete or incompatible run: {path}')
        expected_mode = 'baseline' if name == 'baseline' else 'future'
        if value['mode'] != expected_mode:
            raise ValueError(f'wrong mode: {path}')
        if name != 'baseline' and value['adapter_epoch'] != int(name.split('_')[1]):
            raise ValueError(f'wrong adapter epoch: {path}')
        if any(not math.isfinite(value[key]) for key in (PRIMARY, 'avg_L2_m', 'avg_endpoint_L2_m')):
            raise ValueError(f'non-finite L2: {path}')
        if runs:
            differing = [key for key in PROTOCOL_KEYS if value[key] != runs[0]['metrics'][key]]
            if differing:
                raise ValueError(f'protocol differs at {path}: {differing}')
        runs.append({'name': name, 'metrics_path': str(path), 'metrics': value})
    if require_complete and len(runs) != 11:
        raise ValueError(f'expected baseline + 10 adapters, found {len(runs)}')
    future = sorted((row for row in runs if row['metrics']['mode'] == 'future'),
                    key=lambda row: (row['metrics'][PRIMARY], row['metrics']['adapter_epoch']))
    baseline = next((row for row in runs if row['name'] == 'baseline'), None)
    def choice(key):
        if not future:
            return None
        row = min(future, key=lambda row: (row['metrics'][key], row['metrics']['adapter_epoch']))
        return {'epoch': row['metrics']['adapter_epoch'], 'metric': key,
                'value_m': row['metrics'][key], 'checkpoint': row['metrics']['adapter_checkpoint'],
                'checkpoint_sha256': row['metrics']['adapter_sha256'],
                'metrics_path': row['metrics_path']}
    report = {'status': 'COMPLETE' if len(runs) == 11 else 'IN_PROGRESS',
              'completed_runs': len(runs), 'expected_runs': 11,
              'selection_metric': PRIMARY,
              'selection_definition': 'mean ego endpoint L2 at 1/2/3s, individual masks, after 3-frame cold start',
              'best_val_planning_l2': choice(PRIMARY),
              'best_complete_prefix_l2': choice('avg_L2_m'),
              'best_complete_endpoint_l2': choice('avg_endpoint_L2_m'),
              'baseline': baseline, 'ranked_adapters': future}
    temporary = root / 'selection.json.tmp'
    temporary.write_text(json.dumps(report, indent=2))
    os.replace(str(temporary), str(root / 'selection.json'))
    with (root / 'leaderboard.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'epoch', 'val_L2_1s', 'val_L2_2s', 'val_L2_3s', PRIMARY,
                         'prefix_1s', 'prefix_2s', 'prefix_3s', 'avg_prefix_L2_m',
                         'avg_complete_endpoint_L2_m', 'checkpoint'])
        for row in ([baseline] if baseline else []) + future:
            value = row['metrics']
            writer.writerow([row['name'], value['adapter_epoch'], *value['validation_planning_L2_m'],
                             value[PRIMARY], *value['prefix_l2'], value['avg_L2_m'],
                             value['avg_endpoint_L2_m'], value['adapter_checkpoint']])
    print(json.dumps({key: report[key] for key in
                      ('status', 'completed_runs', 'best_val_planning_l2', 'best_complete_prefix_l2')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.require_complete)
