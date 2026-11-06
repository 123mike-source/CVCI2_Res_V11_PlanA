"""Fresh ten-epoch H20 run with validation after every epoch; dry-run is read-only."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def commands(args, recipe, epoch):
    work = args.output / 'run'
    train = [str(args.python), '-u', '-m', 'torch.distributed.run',
             '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0', '--rdzv-id=v10-refined-train',
             '--nproc_per_node=4', str(ROOT / 'tools/train_v10_refined_h20.py'),
             '--work-dir', str(work), '--data-root', str(args.data_root),
             '--base-checkpoint', str(args.models_dir / 'best_endpoint_1step.pth'),
             '--world-checkpoint', str(args.models_dir / 'MMO_MSO_with_plan_tightly_coupled.pth'),
             '--geobev-checkpoint', str(args.models_dir / 'geobev-r50-nuimage-cbgs.pth'),
             '--world-root', str(args.world_root), '--stop-after-epoch', str(epoch)]
    for name, value in recipe['arguments'].items():
        key = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                train.append(key)
        else:
            train.extend([key] + [str(item) for item in value] if isinstance(value, list) else [key, str(value)])
    if epoch > 1:
        train.extend(['--resume', str(work / 'last.pth')])
    evaluation = [str(args.python), '-u', '-m', 'torch.distributed.run',
        '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0', '--rdzv-id=v10-refined-eval',
        '--nproc_per_node=4', str(ROOT / 'tools/eval_geobev_full_coverage.py'),
        '--adapter-checkpoint', str(work / ('epoch_%02d.pth' % epoch)),
        '--output-dir', str(args.output / ('validation/epoch_%02d' % epoch)),
        '--base-checkpoint', str(args.models_dir / 'best_endpoint_1step.pth'),
        '--world-checkpoint', str(args.models_dir / 'MMO_MSO_with_plan_tightly_coupled.pth'),
        '--data-root', str(args.data_root), '--world-root', str(args.world_root),
        '--seed', str(recipe['arguments']['seed']), '--ddim-steps', '1',
        '--layer-diagnostics',
        '--workers', '2', '--prefetch-factor', '2', '--log-seconds', '30']
    return train, evaluation


def audit_actual(actual, recipe):
    config = actual['config']
    for name, expected in recipe['arguments'].items():
        if config.get(name) != expected:
            raise RuntimeError('Effective training configuration differs: ' + name)
    assert actual['world_trainable_parameters'] == 907712
    assert config['fresh_res'] and not config['warm_start'] and not config['warm_start_geobev']
    assert actual['nominal_effective_batch'] == 48
    assert actual['lr_schedule']['total_updates'] == 5870
    assert actual['lr_schedule']['warmup_updates'] == 500
    groups = actual['training_optimization']['optimizer_groups']
    assert sum(g['parameter_count'] for g in groups if g['name'].startswith('control_')) == 42
    assert actual['fresh_initialization_audit']['prior_res_checkpoint_loaded'] is False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new experiment directory')
    parser.add_argument('--python', type=Path, default=ROOT / 'ENV/bin/python')
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/workspace/dataSet/nuscene'))
    parser.add_argument('--models-dir', type=Path, default=ROOT.parent / 'Models')
    parser.add_argument('--world-root', type=Path, default=ROOT / 'world_configs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for key in ('output', 'python', 'data_root', 'models_dir', 'world_root'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    recipe_file = ROOT / 'configs/v10_refined/train.json'
    recipe = read(recipe_file)
    if args.dry_run:
        print(json.dumps(dict(recipe=recipe, commands={str(epoch): commands(args, recipe, epoch)
            for epoch in (1, recipe['arguments']['epochs'])}), indent=2))
        return
    if args.output.exists():
        raise RuntimeError('Choose a new experiment directory')
    lock = (ROOT / 'ENV/h20_training_launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPUs are occupied; no run was started')
    for name, expected in recipe['original_checkpoints'].items():
        if sha256(args.models_dir / name) != expected:
            raise RuntimeError('Original pretrained checkpoint differs: ' + name)
    args.output.mkdir(parents=True)
    write(args.output / 'training_recipe.json', recipe)
    env = dict(os.environ, **recipe['environment'])
    env.pop('CVCI_DETERMINISTIC_CHECKS', None)
    results = []

    def run(phase, command, log):
        if read(recipe_file) != recipe:
            raise RuntimeError('Recipe changed during this experiment')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(command=command)) + '\n')
            stream.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            write(args.output / 'state.json', dict(phase=phase, pid=os.getpid(), active_pid=process.pid,
                updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat()))
            if process.wait():
                raise RuntimeError(phase + ' failed; see ' + str(log))

    try:
        for epoch in range(1, recipe['arguments']['epochs'] + 1):
            train, evaluation = commands(args, recipe, epoch)
            run('training_epoch%02d' % epoch, train, args.output / 'run/train.log')
            audit_actual(read(args.output / 'run/config.json'), recipe)
            run('validation_epoch%02d' % epoch, evaluation, args.output / ('validation/epoch_%02d.log' % epoch))
            metrics = read(args.output / ('validation/epoch_%02d/metrics.json' % epoch))
            assert metrics['sample_coverage_check'] == 'PASS' and not metrics['partial']
            for key, expected in recipe['validation'].items():
                assert metrics[key] == expected, key
            row = dict(epoch=epoch, checkpoint=str(args.output / ('run/epoch_%02d.pth' % epoch)),
                rolling_4585_L2_m=metrics['retained_original_subset']['avg_L2_m'],
                full_valid_5119_L2_m=metrics['avg_L2_m'])
            assert all(math.isfinite(row[key]) for key in ('rolling_4585_L2_m', 'full_valid_5119_L2_m'))
            results.append(row)
            write(args.output / 'epoch_metrics.json', results)
            write(args.output / 'best.json', min(results, key=lambda value: value['rolling_4585_L2_m']))
        write(args.output / 'state.json', dict(phase='complete', completed_epochs=10))
    except Exception:
        write(args.output / 'state.json', dict(phase='failed', error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
