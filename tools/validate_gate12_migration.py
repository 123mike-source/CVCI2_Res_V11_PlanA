"""Real-checkpoint parity, x0 supervision, joint gradients and rolling regression."""
import argparse
import contextlib
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvci2_gate12 import Gated12Model
from matched_diffusion import MatchedDiTLayer, MatchedX0Planner
from planning_runtime import (DEFAULT_BASE, build_temporal_model, load_planning_checkpoint,
                              planning_model_metadata, source_fingerprints)
from temporal_matched_deform_model import (
    MatchedDeformTemporalModel, matched_deform_state_dict,
    MATCHED_DEFORM_FORMAT, MATCHED_DEFORM_SCOPE, MATCHED_DEFORM_VERSION)


def synthetic_batch(frame, device, target=.1):
    generator = torch.Generator(device='cpu').manual_seed(9000+frame)
    camera = torch.randn(1, 3, 64, 256, generator=generator).to(device)
    lidar = torch.rand(1, 1, 256, 256, generator=generator).to(device)
    states = torch.zeros(1, 30, 5, device=device)
    states[:, 0] = states.new_tensor([3., 2., .1, 2., 4.])
    labels = torch.zeros(1, 30, device=device); labels[:, 0] = 1
    return dict(camera=camera, lidar=lidar,
                status=torch.tensor([[.2, .1, .05, 1., .02, .1, .2, .3]], device=device),
                command=torch.tensor([[0., 1., 0.]], device=device),
                agent_states=states, agent_labels=labels,
                world_image=torch.full((1, 6, 3, 8, 8), float(frame), device=device),
                projection=torch.eye(4, device=device).expand(1, 6, 4, 4),
                world_status=torch.zeros(1, 10, device=device),
                lidar_to_ego=torch.eye(4, dtype=torch.float64, device=device)[None],
                data_index=torch.tensor([frame], device=device),
                trajectory_delta=torch.full((1, 6, 2), target, device=device),
                mask=torch.ones(1, 6, device=device),
                meta=[dict(token=str(frame), scene_token='test-scene', timestamp=frame*.5,
                           T_global=np.eye(4))])


class RecordingWorld(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones((), device=device))
        # Smooth, nonconstant grids make offset task gradients observable.
        axis = torch.linspace(-1., 1., 200, device=device)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        channels = torch.arange(256, device=device)[:, None, None]
        features = (torch.sin(x[None]*(channels+1)/15) + torch.cos(y[None]*(channels+1)/17))
        self.register_buffer('features', features[None, None].repeat(1, 6, 1, 1, 1), persistent=False)
        self.calls = []

    def forward(self, images, projections, metas, prior, statuses):
        self.calls.append(dict(images=list(images.shape), projections=list(projections.shape),
                               frames=images[0, :, 0, 0, 0, 0].tolist(),
                               prior=list(prior.shape), statuses=list(statuses.shape),
                               grad_enabled=torch.is_grad_enabled(), training=self.training))
        return self.features.expand(len(images), -1, -1, -1, -1)


def validate(device='cuda'):
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(2026)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(status='RUNNING', device=device, checks={})

    def passed(name, value=True):
        report['checks'][name] = value
        print('PASS', name, value, flush=True)

    original_state = torch.load(DEFAULT_BASE, map_location='cpu')['model']
    reference = Gated12Model()
    reference.load_state_dict(original_state, strict=True)
    model = MatchedDeformTemporalModel(enable_world=False)
    state = model.base.state_dict()
    assert all(torch.equal(state[name], value) for name, value in original_state.items())
    assert set(state)-set(original_state) == {k for k in state if '.future_' in k}
    assert len(model.base.planner.layers) == 12
    assert sum(layer.interaction for layer in model.base.planner.layers) == 6
    assert all(parameter.requires_grad for parameter in model.base.parameters())
    passed('real_gate12_checkpoint_all_shared_tensors_identical', len(original_state))
    del original_state, state

    # Also compare the independently imported original checkout, when available.
    original_checkout = ROOT.parent/'CVCI2'
    if original_checkout.is_dir():
        sys.path.insert(0, str(original_checkout))
        original_class = importlib.import_module('gate_model').Gated12Model
        original = original_class()
        original.load_state_dict(reference.state_dict(), strict=True)
        reference = original
        sys.path.remove(str(original_checkout))
        passed('independent_original_checkout_import')
    reference = reference.to(device).eval()
    model = model.to(device).eval()
    batch = synthetic_batch(0, device)
    noise = torch.randn(1, 6, 2, device=device)
    precisions = ['fp32', 'bf16'] if device == 'cuda' else ['fp32']
    maximum = {}
    for precision in precisions:
        cast = (lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)) if precision == 'bf16' else contextlib.nullcontext
        with torch.no_grad(), cast():
            args = [batch[k] for k in ('camera', 'lidar', 'status', 'command')]
            original_conditions = reference.conditions(*args)
            conditions = model.base.conditions(*args)
            assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(original_conditions, conditions))
            ego, agents, dense, status, _, canbus = conditions
            step = torch.tensor([99], device=device)
            a = reference.planner.predict(noise, step, ego, agents, dense, status, canbus)
            b = model.base.planner.predict(noise, step, ego, agents, dense, status, canbus=canbus)
            assert len(a) == len(b) == 12
            assert all(torch.allclose(x,y,atol=1e-5,rtol=1e-5) for x,y in zip(a,b))
            for steps in [1,2]:
                expected = reference.sample(*args, initial_noise=noise, inference_steps=steps)
                actual = model.base.planner.sample(1, ego, agents, dense, status,
                    initial_noise=noise, inference_steps=steps, canbus=canbus)
                assert torch.isfinite(actual).all()
                error = float((actual-expected).abs().max())
                assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-5), (precision, steps, error)
                maximum[precision+'_'+str(steps)+'step_max_error'] = error
    passed('disabled_future_reference_parity', maximum)

    ego,agents,dense,status,canbus = [x.float() for x in (ego,agents,dense,status,canbus)]
    planner = model.base.planner
    target = torch.full((1, 6, 2), .3, device=device)
    mask = torch.tensor([[1,1,1,1,0,0]], dtype=torch.float32, device=device)
    clean = planner.normalize(target)
    fake = [(clean + (index+1)*.01).detach().requires_grad_() for index in range(12)]
    with patch.object(planner, 'predict', return_value=fake):
        terms = planner.loss(target, mask, ego, agents, dense, status, canbus=canbus, return_details=True)
    manual = torch.tensor([((i+1)*.01)**2 for i in range(12)], device=device).mean()
    assert torch.allclose(terms['total'], manual, atol=1e-7, rtol=1e-5)
    assert 'trajectory' not in terms
    gradients = torch.autograd.grad(terms['total'], fake)
    scaled = [g.abs().sum()/(i+1) for i,g in enumerate(gradients)]
    assert all(torch.allclose(scaled[0], g, atol=1e-7, rtol=1e-5) for g in scaled)
    passed('equal_12_block_x0_supervision_without_endpoint_huber', float(terms['total']))

    # A disabled future path must ignore even nonzero residual gates and tokens.
    tokens = dict(tokens=torch.randn(1, 5, 256, device=device), valid=torch.ones(1, 5, dtype=torch.bool, device=device),
                  log_prior=torch.zeros(1, 5, device=device), map_ids=torch.arange(5, device=device),
                  waypoint_ids=torch.arange(5, device=device))
    gates=[layer.future_gate for layer in planner.layers if isinstance(layer,MatchedDiTLayer)]
    assert all(float(g)==0 for g in gates)
    wake_loss=planner.loss(target,mask,ego,agents,dense,status,canbus=canbus,
                           future_tokens=tokens,future_mask=torch.ones(1,dtype=torch.bool,device=device))
    wake_gradients=torch.autograd.grad(wake_loss,gates)
    assert all(torch.isfinite(g).all() for g in wake_gradients)
    assert sum(float(g.abs()) for g in wake_gradients)>0
    passed('zero_initialized_future_gates_receive_task_gradients')
    with torch.no_grad():
        for layer in planner.layers:
            if isinstance(layer, MatchedDiTLayer):layer.future_gate.fill_(.2)
        first = planner.predict(noise, step, ego, agents, dense, status, canbus=canbus)[-1]
        second = planner.predict(noise, step, ego, agents, dense, status, canbus=canbus,
            future_tokens=tokens, future_mask=torch.zeros(1, dtype=torch.bool, device=device))[-1]
        assert torch.equal(first, second)
    passed('future_mask_preserves_gate12_output')
    del reference, original_conditions, conditions, a, b, fake, gradients, tokens
    gc.collect()
    if device == 'cuda':torch.cuda.empty_cache()

    world = RecordingWorld(device)
    model.world = world
    model.enable_world = True
    for parameter in world.parameters():parameter.requires_grad_(False)
    model.eval()

    def stream(target_value):
        model.reset();flags=[];last=None
        with torch.no_grad():
            for frame in range(4):
                result = model(synthetic_batch(frame, device, target_value), compute_loss=False,
                               initial_noise=noise, skip_warmup=True)
                flags.append((bool(result['ready'][0]),bool(result['prior_valid'][0])))
                if result['prediction'] is not None:last=result['prediction'].clone()
        return flags,last

    flags, first = stream(.1)
    assert flags == [(False,False),(False,False),(True,False),(True,True)]
    assert len(world.calls)==1 and world.calls[0]['frames']==[1.,2.,3.]
    assert world.calls[0]['images'][:3]==[1,3,6]
    assert not world.calls[0]['grad_enabled'] and not world.calls[0]['training']
    _, second = stream(99.)
    assert torch.equal(first, second)
    passed('rolling_cold_start_previous_prediction_and_no_future_gt', world.calls[0])

    model.train()
    model.fusion_config['future_dropout'] = 0.
    assert model.base.backbone.training and model.base.decoder.training
    assert not world.training and all(not p.requires_grad for p in world.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    cast = torch.cuda.amp.autocast(dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()
    with cast:
        result = model(synthetic_batch(4, device), monitor=True, initial_noise=noise)
    expected = result['loss_x0']+.2*result['loss_agent_class']+.2*result['loss_agent_box']+.1*result['loss_occupancy']
    assert torch.allclose(result['loss'], expected)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    required = ['base.backbone.image.', 'base.backbone.lidar.', 'base.bev_down.',
                'base.status.', 'base.route.', 'base.decoder.', 'base.dense_proj.',
                'base.agent_head.', 'base.occupancy_head.', 'base.spatial_gate.canbus_mlp.',
                'base.spatial_gate.navi_embedding.', 'base.planner.canbus_concat.',
                'base.planner.action.', 'base.planner.layers.1.future_attn.',
                'future_encoder.value_projection.', 'future_encoder.offset.']
    norms = {}
    for prefix in required:
        gradients = [p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), prefix
        norm = float(torch.stack([g.float().norm() for g in gradients]).norm())
        assert norm>0, (prefix,norm)
        norms[prefix] = norm
    assert all(p.grad is None for p in world.parameters())
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    optimizer.step()
    passed('joint_auxiliary_losses_and_nonzero_task_gradients', norms)

    # Cold-start perception losses must survive a completely masked planning loss.
    model.reset();optimizer.zero_grad(set_to_none=True)
    with (torch.cuda.amp.autocast(dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()):
        cold = model(synthetic_batch(0, device), initial_noise=noise)
    assert not cold['ready'].any() and float(cold['loss_x0']) == 0
    assert cold['loss_auxiliary'] > 0
    cold['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.base.agent_head.parameters())
    passed('cold_start_frames_train_perception')
    optimizer.zero_grad(set_to_none=True)

    snapshot = matched_deform_state_dict(model)
    assert any(k.startswith('base.backbone.') for k in snapshot)
    assert any(k.endswith('running_mean') for k in snapshot)
    assert not any(k.startswith('world.') for k in snapshot)
    checkpoint = dict(format=MATCHED_DEFORM_FORMAT,trainable=snapshot,
        metadata=dict(architecture_version=MATCHED_DEFORM_VERSION,training_scope=MATCHED_DEFORM_SCOPE,
                      fusion_config=model.fusion_config,sampler_config=model.sampler_config,initialization_seed=2026))
    restored = build_temporal_model(checkpoint, base_checkpoint='/does/not/exist', enable_world=False)
    load_planning_checkpoint(restored,checkpoint)
    restored_state = matched_deform_state_dict(restored)
    assert set(restored_state)==set(snapshot)
    assert all(torch.equal(restored_state[k],v) for k,v in snapshot.items())
    passed('full_joint_checkpoint_round_trip_including_bn_without_base_file', len(snapshot))
    for bad in [dict(format='cvci2res_matched_deform_denoise_v5',metadata=dict(architecture_version=5)),
                dict(format=MATCHED_DEFORM_FORMAT,metadata=dict(architecture_version=5))]:
        try:build_temporal_model(bad,enable_world=False)
        except ValueError:pass
        else:raise AssertionError('Retired epsilon checkpoint was accepted')
    passed('old_epsilon_checkpoints_rejected')
    sources = source_fingerprints(ROOT)
    assert 'cvci2_gate12/gate_model.py' in sources and 'data_paths.py' in sources
    assert not any(k.startswith(('ENV/','.python/','.env-')) for k in sources)
    passed('source_snapshot_excludes_local_environment', len(sources))
    report.update(status='PASS',architecture=planning_model_metadata(model))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--output',default=str(ROOT/'ENV/gate12_migration_report.json'))
    args=parser.parse_args()
    result=validate(args.device)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
