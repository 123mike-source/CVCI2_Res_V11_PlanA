"""Real-input equivalence, gradient and checkpoint tests for motion gating."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def feature_contract():
    from motion_gate import observed_motion_features, zero_motion_projection
    status = torch.zeros(2, 3, 10, device='cuda')
    status[0, :, 6] = status.new_tensor([0., .2, .5])
    status[0, :, :2] = status.new_tensor([1., 2.])
    status[0, :, 5] = status.new_tensor([.1, .2, .3])
    times = [[{'timestamp': x} for x in [10., 10.5, 11.]],
             [{'timestamp': x} for x in [10., 10., 10.]]]
    prior = torch.zeros(2, 6, 2, device='cuda')
    features, valid = observed_motion_features(status, times, prior, torch.ones(2, device='cuda', dtype=torch.bool))
    expected = features.new_tensor([0.,0.,.2/15,0.,.5/15,0.,1/5,2/5,1/5,2/5,1/5,2/5,
                                    .1,.2,.3,.4/5,.6/5,-.5/15,0.,1/6])
    torch.testing.assert_close(features[0, 0], expected)
    assert valid.tolist() == [True, False]
    assert torch.count_nonzero(features[1]).item() == 0
    changed = status.clone(); changed[0, 1, 6] = float('nan')
    invalid, valid = observed_motion_features(changed, times, prior, torch.ones(2, device='cuda', dtype=torch.bool))
    assert not valid.any() and torch.isfinite(invalid).all() and not invalid.any()
    before = torch.get_rng_state().clone(); projection = zero_motion_projection()
    assert torch.equal(before, torch.get_rng_state()) and projection.weight.count_nonzero() == 0
    return {'status':'PASS','dimensions':20,'history_order':'oldest_to_newest',
            'actual_dt_used':True,'invalid_history_zeroed':True,'initialization_rng_preserved':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.monotonic()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(2026)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    from motion_gate import (DEFAULT_MOTION_GATE_CONFIG, MOTION_GATE_FORMAT, MOTION_GATE_VERSION, warm_start_v6)
    from temporal_matched_deform_model import (MatchedDeformTemporalModel, matched_deform_state_dict,
                                               MATCHED_DEFORM_SCOPE)
    from temporal_dataset import TemporalNuScenesDataset, temporal_collate, to_device
    from data_paths import DEFAULT_DATA_ROOT
    from benchmark_h20_training import sequence
    from train_h20 import configure_world_batch
    from training_optimization import optimizer_groups
    from planning_runtime import build_temporal_model,load_planning_checkpoint
    result={'feature_contract':feature_contract()}
    data=TemporalNuScenesDataset(DEFAULT_DATA_ROOT,str(ROOT/'ENV/h20_benchmark_subset.pkl'))
    frames=list(sequence(data));assert len(frames)>=4
    frames=frames[:8]
    settings=dict(fusion_config={'future_dropout':.1},sampler_config={
        'points_per_map':[36,25,16,9,9],'window_m':[6.,3.],'offset_mode':'learned'})
    base=MatchedDeformTemporalModel(**settings).cuda().eval();configure_world_batch(base,1)
    gate=MatchedDeformTemporalModel(**settings,world=base.world,motion_gate_config=DEFAULT_MOTION_GATE_CONFIG).cuda().eval()
    old_audit=warm_start_v6(base,args.checkpoint);new_audit=warm_start_v6(gate,args.checkpoint)
    assert new_audit['new_parameter_count']==15360 and len(new_audit['new_zero_tensors'])==6
    assert not any(p.requires_grad for p in gate.world.parameters())
    result['initialization']={'status':'PASS','base':old_audit,'motion':new_audit}
    groups,manifest=optimizer_groups(gate,profile='differential',lr=1e-4)
    selected={id(p):g['lr'] for g in groups for p in g['params']}
    projections=[p for n,p in gate.named_parameters() if n.endswith('.future_motion_projection.weight')]
    assert all(selected[id(p)]==1e-4 for p in projections)
    result['optimizer']={'status':'PASS','new_projection_tensors':len(projections),'new_lr':1e-4}
    records=[]
    for precision in ['fp32','bf16']:
        base.reset();gate.reset();max_prediction=0.;max_loss=0.;active_frames=0
        for i,sample in enumerate(frames):
            batch=to_device(temporal_collate([sample]),torch.device('cuda'))
            outputs=[]
            for model in [base,gate]:
                torch.manual_seed(2026+i)
                with torch.no_grad(),torch.cuda.amp.autocast(enabled=precision=='bf16',dtype=torch.bfloat16):
                    outputs.append(model(batch,monitor=True))
            a,b=outputs
            assert torch.equal(a['ready'],b['ready']) and torch.equal(a['future_used'],b['future_used'])
            torch.testing.assert_close(a['prediction'],b['prediction'],atol=2e-5,rtol=1e-5)
            torch.testing.assert_close(a['loss'],b['loss'],atol=1e-6,rtol=1e-5)
            max_prediction=max(max_prediction,float((a['prediction']-b['prediction']).abs().max()))
            max_loss=max(max_loss,float((a['loss']-b['loss']).abs()))
            active_frames+=int(b['future_used'].sum())
            if b['future_used'].any():
                assert gate.last_motion_features.shape==(1,6,20)
                assert torch.isfinite(gate.last_motion_features).all()
            del outputs,a,b
        assert active_frames>0
        records.append({'precision':precision,'frames':len(frames),'active_frames':active_frames,
                        'max_prediction_abs_m':max_prediction,'max_loss_abs':max_loss})
    result['zero_projection_equivalence']={'status':'PASS','records':records}
    with torch.no_grad():
        for p in projections:p.fill_(.03)
    base.reset();gate.reset();masked_diff=0.
    for i,sample in enumerate(frames[:4]):
        batch=to_device(temporal_collate([sample]),torch.device('cuda'))
        with torch.no_grad():
            a=base(batch,compute_loss=False,disable_future=True)
            b=gate(batch,compute_loss=False,disable_future=True)
        torch.testing.assert_close(a['prediction'],b['prediction'],atol=2e-5,rtol=1e-5)
        masked_diff=max(masked_diff,float((a['prediction']-b['prediction']).abs().max()))
    with torch.no_grad():
        for p in projections:p.zero_()
    result['future_disabled_invariance']={'status':'PASS','max_prediction_abs_m':masked_diff}
    gate.reset();gate.eval()
    with torch.no_grad():
        for sample in frames[:3]:
            batch=to_device(temporal_collate([sample]),torch.device('cuda'));gate(batch,compute_loss=False)
    gate.train();gate.fusion_config['future_dropout']=0.
    batch=to_device(temporal_collate([frames[3]]),torch.device('cuda'));gate.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):out=gate(batch)
    assert out['future_used'].all()
    out['loss_x0'].backward()
    norms=[float(p.grad.float().norm()) for p in projections]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in projections)
    opt=torch.optim.AdamW(projections,lr=1e-4,fused=True);opt.step()
    assert all(p.count_nonzero()>0 for p in projections)
    result['gradient_flow']={'status':'PASS','projection_gradient_norms':norms,
                              'diagnostic_optimizer_updates':1,'production_checkpoint_modified':False}
    gate.fusion_config['future_dropout']=.1;gate.eval()
    metadata={'architecture_version':MOTION_GATE_VERSION,'training_scope':MATCHED_DEFORM_SCOPE,
              'initialization_seed':2026,'fusion_config':gate.fusion_config,'sampler_config':gate.sampler_config,
              'motion_gate_config':gate.motion_gate_config}
    checkpoint={'format':MOTION_GATE_FORMAT,'metadata':metadata,'trainable':matched_deform_state_dict(gate),'epoch':0}
    target=Path(args.output).parent/'v7_roundtrip_diagnostic.pth';torch.save(checkpoint,target)
    reread=torch.load(target,map_location='cpu')
    restored=build_temporal_model(reread,world=base.world).cuda().eval();load_planning_checkpoint(restored,reread)
    assert all(torch.equal(restored.state_dict()[n].cpu(),v) for n,v in checkpoint['trainable'].items())
    result['v7_roundtrip']={'status':'PASS','format':reread['format'],'all_tensors_equal':True}
    old=torch.load(args.checkpoint,map_location='cpu')
    old_model=build_temporal_model(old,world=base.world).cuda().eval();load_planning_checkpoint(old_model,old)
    assert old_model.motion_gate_config is None
    result['v6_factory_compatibility']={'status':'PASS','legacy_motion_projection_absent':True}
    result.update(status='PASS',elapsed_seconds=time.monotonic()-started,
                  test_cudnn_benchmark=False,production_cudnn_benchmark=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','output':args.output,'new_parameters':15360,
        'equivalence':records,'gradient_norm_min':min(norms),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
