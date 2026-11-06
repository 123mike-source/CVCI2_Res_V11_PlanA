"""Check the reviewed training recipe on real data; outputs are not accuracy results."""
import argparse
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import torch


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(out):
    from matched_diffusion import MatchedX0Planner
    from training_optimization import canonical_layer_weights, diagnostic_values
    old = imported('old_matched', out / 'source_before/matched_diffusion.py')

    class Fake:
        train_timesteps = 100
        alpha = torch.linspace(.999, .00000025, 100)
        training_t99_fraction = 0.
        loss_diagnostics_enabled = False
        layer_loss_weights = None
        normalize = staticmethod(lambda x: x)
        denormalize = staticmethod(lambda x: x)

        def predict(self, value, timestep, *args, **kwargs):
            self.timestep, self.noisy = timestep, value
            self.last_training_candidate_positions = torch.zeros_like(value)
            return [value * self.scales[i] for i in range(12)]

    delta = torch.linspace(-.5, .5, 8 * 6 * 2).reshape(8, 6, 2)
    mask = torch.ones(8, 6)
    mask[0] = 0

    def run(function, diagnostics=False, fraction=0., weights=None):
        fake = Fake()
        fake.scales = torch.linspace(.1, 1.2, 12).requires_grad_()
        fake.loss_diagnostics_enabled, fake.training_t99_fraction = diagnostics, fraction
        fake.layer_loss_weights = weights
        torch.manual_seed(901)
        result = function(fake, delta, mask, None, None, None, None, return_details=True)
        result['x0'].backward()
        return fake, result, fake.scales.grad, torch.get_rng_state()

    a = run(old.MatchedX0Planner.loss)
    b = run(MatchedX0Planner.loss)
    c = run(MatchedX0Planner.loss, diagnostics=True)
    assert torch.equal(a[1]['x0'], b[1]['x0']) and torch.equal(a[2], b[2])
    assert torch.equal(b[1]['x0'], c[1]['x0']) and torch.equal(b[2], c[2])
    assert torch.equal(a[3], b[3]) and torch.equal(b[3], c[3])
    weights = canonical_layer_weights([1] * 8 + [2, 2, 2, 6])
    d = run(MatchedX0Planner.loss, diagnostics=True, fraction=.5, weights=weights)
    assert torch.allclose(d[1]['x0'], (d[1]['layer_losses'] * torch.tensor(weights)).sum())
    counts = diagnostic_values(d[1]['diagnostics'])
    assert sum(counts['valid_samples_' + key] for key in ('t00_24', 't25_49', 't50_74', 't75_99')) == 7
    e = run(MatchedX0Planner.loss, fraction=1.)
    assert bool((e[0].timestep == 99).all())
    # Verify the actual override sampler, including its uniform component.
    large = Fake()
    large.scales = torch.ones(12)
    large.training_t99_fraction = .5
    torch.manual_seed(902)
    MatchedX0Planner.loss(large, torch.zeros(20000, 6, 2), torch.ones(20000, 6),
                         None, None, None, None)
    p99 = float((large.timestep == 99).float().mean())
    assert abs(p99 - .505) < .015, p99
    return dict(legacy_loss_and_gradients_exact=True, diagnostics_preserve_rng=True,
                weighted_loss_checked=True, effective_p99_observed=p99, layer_weights=weights)


def dataset_and_color(out):
    from PIL import Image
    from geobev_encoder.dataset import GeoBEVNuScenesDataset
    from geobev_encoder.model import GeoBEVRollingState, GEO_FIELDS
    from geobev_encoder.augmentation import color_config
    from temporal_dataset import temporal_collate, to_device
    from training_optimization import exact_state_equal
    data = GeoBEVNuScenesDataset('/mnt/workspace/dataSet/nuscene',
        '/mnt/workspace/dataSet/nuscene/nuscenes_infos_train.pkl')
    scene = next(indices[start:start + 6] for indices in data.scenes.values()
        for start in range(len(indices) - 5)
        if all(abs((data.infos[b]['timestamp'] - data.infos[a]['timestamp']) / 1e6 - .5) <= .1
               for a, b in zip(indices[start:start + 5], indices[start + 1:start + 6]))
        and all(bool(torch.as_tensor(data.infos[i]['gt_ego_fut_masks']).all())
                for i in indices[start:start + 6]))
    old_temporal = imported('old_temporal', out / 'source_before/temporal_dataset.py')
    old_geo = imported('geobev_encoder.old_dataset', out / 'source_before/geobev_encoder/dataset.py')
    old_geo.GeoBEVNuScenesDataset._world_views = old_temporal.TemporalNuScenesDataset._world_views
    legacy = object.__new__(old_geo.GeoBEVNuScenesDataset)
    legacy.__dict__ = data.__dict__.copy()
    count = [0]
    original_resize = Image.Image.resize

    def counted(*args, **kwargs):
        count[0] += 1
        return original_resize(*args, **kwargs)

    Image.Image.resize = counted
    try:
        previous = legacy[scene[0]]
        previous_resizes = count[0]
        count[0] = 0
        current = data[scene[0]]
        current_resizes = count[0]
    finally:
        Image.Image.resize = original_resize
    assert exact_state_equal(previous, current), 'Shared preprocessing changed a tensor/calibration/label'
    assert current_resizes == 6 and previous_resizes == 12
    samples = [current] + [data[i] for i in scene[1:6]]
    for index in scene[1:3]:
        assert exact_state_equal(legacy[index], data[index])
    stream = GeoBEVRollingState()
    stream.configure_color(color_config(1., .1), 7788)
    stream.augmentation_enabled = True
    rng = torch.get_rng_state().clone()
    factors = []
    for sample in samples[:3]:
        batch = to_device(temporal_collate([sample]), torch.device('cpu'))
        raw = batch['geobev_image'].clone()
        stream.update(batch)
        assert torch.equal(raw, batch['geobev_image']), 'Augmentation mutated the raw input'
        assert torch.equal(stream.slots[0]['frames'][-1]['image'], batch['world_image'][0])
        factors.append(stream.color.slots[0]['factors'])
    assert factors[0] == factors[1] == factors[2]
    assert torch.equal(rng, torch.get_rng_state()), 'Color RNG perturbed model noise RNG'
    saved = copy.deepcopy(stream.state_dict())
    restored = GeoBEVRollingState()
    restored.configure_color(color_config(1., .1), 7788)
    restored.augmentation_enabled = True
    restored.load_state_dict(saved, data, torch.device('cpu'))
    assert exact_state_equal(stream.state_dict(), restored.state_dict())
    for left, right in zip(stream.slots[0]['frames'], restored.slots[0]['frames']):
        for key in ('image',) + GEO_FIELDS:
            assert torch.equal(left[key], right[key]), key
    for sample in samples[3:5]:
        batch = temporal_collate([sample])
        assert exact_state_equal(stream.update(batch), restored.update(batch))
    new_scene = copy.deepcopy(samples[5])
    new_scene['meta']['scene_token'] += ':new'
    batch = temporal_collate([new_scene])
    assert exact_state_equal(stream.update(batch), restored.update(batch))
    assert exact_state_equal(stream.state_dict(), restored.state_dict())
    return data, samples, dict(raw_preprocessing_exact=True, old_resize_calls=previous_resizes,
        shared_resize_calls=current_resizes, color_rng_isolated=True, same_scene_factors=factors[0],
        augmented_history_exact_restore=True, next_scene_rng_exact_restore=True, world_image_unchanged=True)


def build_and_partition():
    from geobev_encoder.model import GeoBEVTemporalModel
    from geobev_encoder.checkpoint import audit_fresh_geobev
    from world_models.world_partial import WORLD_TUNING_CONFIG
    from world_models.adaptive_sampler import ADAPTIVE_SAMPLER_CONFIG
    from motion_gate import INTEGRATED_MOTION_GATE_CONFIG
    from planning_runtime import DEFAULT_BASE, DEFAULT_WORLD, DEFAULT_WORLD_ROOT
    from training_optimization import configure_training, optimizer_groups
    from candidate_interaction import activate, config_for_mode
    model = GeoBEVTemporalModel(DEFAULT_BASE, DEFAULT_WORLD, DEFAULT_WORLD_ROOT,
        seed=2026, initialization_seed=2026, fusion_config={'future_dropout': .1},
        sampler_config=ADAPTIVE_SAMPLER_CONFIG, motion_gate_config=INTEGRATED_MOTION_GATE_CONFIG,
        world_tuning_config=WORLD_TUNING_CONFIG,
        geobev_checkpoint=ROOT.parent / 'Models/geobev-r50-nuimage-cbgs.pth')
    audit = audit_fresh_geobev(model, DEFAULT_BASE, DEFAULT_WORLD, 2026)
    activate(model, config_for_mode('full'))
    config = json.loads((ROOT / 'configs/v10_refined/train.json').read_text())['arguments']
    args = SimpleNamespace(**dict(config, no_world_checkpoint=False))
    configure_training(model, args)
    groups, manifest = optimizer_groups(model, 'differential',
        lr=args.lr, image_lr=args.image_lr, pretrained_lr=args.pretrained_lr,
        planner_lr=args.planner_lr, future_lr=args.future_lr, offset_lr=args.offset_lr,
        world_lr=args.world_lr, gate_lr=args.gate_lr, distance_lr=args.distance_lr,
        current_point_lr=args.current_point_lr, weight_decay=args.weight_decay,
        new_weight_decay=args.new_weight_decay)
    controls = {g['name']: g for g in manifest if g['name'].startswith('control_')}
    assert {n: g['parameter_count'] for n, g in controls.items()} == {
        'control_future_gate': 6, 'control_future_distance': 6, 'control_current_points': 30}
    assert all(g['weight_decay'] == 0 for g in controls.values())
    assert sum(p.numel() for p in model.world.parameters() if p.requires_grad) == 907712
    assert all(p.requires_grad == model.is_trainable(n) for n, p in model.named_parameters())
    for group in manifest:
        for name, reason in group['decay_reasons'].items():
            if reason in ('normalization', 'embedding', 'bias', 'gate_module', 'control_or_vector',
                          'protect_hard_foreground_rows', 'independent_control'):
                assert group['weight_decay'] == 0, name
            elif reason == 'new_matrix':
                assert group['weight_decay'] == .01, name
    return model, groups, audit, manifest


def real_steps(model, groups, samples):
    from temporal_dataset import temporal_collate, to_device
    from training_optimization import ControlUpdateMonitor
    from tools.train_h20 import configure_world_batch
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    configure_world_batch(model, 1)
    old_names = {id(p): n for n, p in model.named_parameters()}
    group_names = [[old_names[id(p)] for p in group['params']] for group in groups]
    model.to(device).train()
    current = dict(model.named_parameters())
    for group, names in zip(groups, group_names):
        group['params'] = [current[name] for name in names]
    model.world.checkpoint_future = True
    # Ensure at least two active future steps, without changing the recipe on disk.
    model.fusion_config['future_dropout'] = 0.
    optimizer = torch.optim.AdamW(groups, fused=True)
    control = ControlUpdateMonitor(model)
    rows = []
    for index, sample in enumerate(samples):
        batch = to_device(temporal_collate([sample]), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(batch, monitor=True)
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        assert all(p.grad is None for n, p in model.named_parameters()
                   if n.startswith('world.') and not p.requires_grad)
        tuned = [p.grad.detach().float().norm() for p in model.world.parameters() if p.requires_grad and p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        control.before_step()
        optimizer.step()
        control.after_step(True)
        rows.append(dict(frame=index + 1, total=float(output['loss']), x0_weighted=float(output['loss_x0']),
            x0_unweighted=float(output['loss_x0_unweighted']), future_used=bool(output['future_used'][0]),
            ready=bool(output['ready'][0]), prior_valid=bool(output['prior_valid'][0]),
            target_valid_points=int(batch['mask'].sum()),
            future_gates=[float(layer.future_gate.detach()) for layer in model.base.planner.layers if hasattr(layer, 'future_gate')],
            world_tuned_gradient=float(torch.stack(tuned).norm()) if tuned else 0.))
        print('REAL_STEP ' + json.dumps(rows[-1]), flush=True)
    assert any(row['world_tuned_gradient'] > 0 for row in rows if row['future_used'])
    assert all(torch.isfinite(value).all() for value in model.last_monitor.values())
    model.fusion_config['future_dropout'] = .1
    return dict(status='PASS', updates=len(rows), frames=rows,
                peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                frozen_world_gradients_absent=True, partial_world_gradient_nonzero=True,
                accuracy_experiment=False)


def checkpoint_checks(model, out):
    from candidate_interaction import CANDIDATE_FORMAT
    from planning_runtime import planning_model_metadata, build_temporal_model, load_planning_checkpoint
    from temporal_matched_deform_model import matched_deform_state_dict
    from training_optimization import exact_state_equal
    metadata = dict(architecture_version=10, initialization_seed=2026,
                    fusion_config=model.fusion_config, **planning_model_metadata(model))
    compact = dict(format=CANDIDATE_FORMAT, epoch=0,
                   trainable=matched_deform_state_dict(model), metadata=metadata)
    torch.save(compact, out / 'verification_only.pth')
    restored = build_temporal_model(compact, enable_world=False, seed=2026)
    load_planning_checkpoint(restored, compact)
    assert exact_state_equal(matched_deform_state_dict(restored), compact['trainable'])
    assert not restored.stream.augmentation_enabled
    assert planning_model_metadata(restored)['training_conditioning'] == metadata['training_conditioning']
    del restored, compact
    gc.collect()
    best = json.loads((ROOT / 'experiments/h20_v10_fresh_base10_20261102_01/best.json').read_text())
    old = torch.load(best['checkpoint'], map_location='cpu')
    restored = build_temporal_model(old, enable_world=False, seed=2026)
    load_planning_checkpoint(restored, old)
    assert exact_state_equal(matched_deform_state_dict(restored), old['trainable'])
    assert restored.color_augmentation_config is None
    assert getattr(restored.base.planner, 'layer_loss_weights', None) is None
    return dict(refined_factory_exact_restore=True, old_v10_factory_exact_restore=True,
                evaluation_color_disabled=True, training_contract_preserved=True)


@torch.no_grad()
def inference_equivalence(model, out):
    from matched_diffusion import MatchedDiTLayer
    old = imported('old_inference_matched', out / 'source_before/matched_diffusion.py')
    old.MatchedDiTLayer = MatchedDiTLayer
    planner = model.base.planner.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(881)
        args = (torch.randn(1, 6, 2), torch.full((1,), 99, dtype=torch.long),
                torch.randn(1, 1, 256), torch.randn(1, 40, 256),
                torch.randn(1, 256, 64, 64), torch.randn(1, 1, 256))
        canbus = torch.randn(1, 1, 256)
        expected = old.MatchedX0Planner.predict(planner, *args, canbus=canbus)
        actual = planner.predict(*args, canbus=canbus)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    return dict(all_twelve_layer_outputs_exact=True, maximum_difference=0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = {'status': 'running'}

    def save(phase):
        result['phase'] = phase
        (out / 'real_checks.json').write_text(json.dumps(result, indent=2))
        print('PHASE ' + phase, flush=True)

    save('policy')
    result['policy'] = policy(out)
    save('dataset_color')
    data, samples, result['preprocessing_and_color'] = dataset_and_color(out)
    save('original_pretraining_and_partition')
    model, groups, result['fresh_audit'], manifest = build_and_partition()
    result['inference_equivalence'] = inference_equivalence(model, out)
    (out / 'optimizer_manifest.json').write_text(json.dumps(manifest, indent=2))
    result['optimizer_groups'] = [{key: value for key, value in row.items()
        if key not in ('parameter_names', 'decay_reasons')} for row in manifest]
    save('real_six_frame_bf16')
    result['real_steps'] = real_steps(model, groups, samples)
    save('checkpoint_factories')
    result['checkpoints'] = checkpoint_checks(model, out)
    result['status'] = 'PASS'
    save('complete')


if __name__ == '__main__':
    main()
