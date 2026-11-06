"""Check legacy loss/RNG preservation and real frozen-world readout equivalence."""
import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import random
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training_optimization import optimizer_groups, restore_fused_optimizer_devices, LOSS_DIAGNOSTIC_KEYS


def write(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(path)


def policy_checks(baseline_source, output):
    from matched_diffusion import MatchedX0Planner
    from temporal_matched_deform_model import MatchedDeformTemporalModel
    spec=importlib.util.spec_from_file_location('original_matched_diffusion',str(Path(baseline_source)/'matched_diffusion.py'))
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    delta=torch.arange(16*6*2,dtype=torch.float32).view(16,6,2)/400
    mask=torch.ones(16,6);mask[0,:]=0;mask[1,3:]=0
    def run(loss_fn,fraction=0.,diagnostics=False):
        parameter=torch.nn.Parameter(torch.tensor(.02));seen={}
        def predict(noisy,timestep,*args,**kwargs):
            seen['timestep']=timestep.detach().clone()
            return [noisy*.1+parameter+i*.001 for i in range(12)]
        fake=SimpleNamespace(normalize=lambda value:value,train_timesteps=100,
            alpha=torch.linspace(.99,.00001,100),predict=predict,
            training_t99_fraction=fraction,loss_diagnostics_enabled=diagnostics)
        torch.manual_seed(723)
        terms=loss_fn(fake,delta,mask,None,None,None,None,return_details=True)
        gradient=torch.autograd.grad(terms['total'],parameter)[0]
        return terms,gradient,torch.get_rng_state(),seen['timestep']
    old=run(original.MatchedX0Planner.loss)
    new=run(MatchedX0Planner.loss)
    observed=run(MatchedX0Planner.loss,diagnostics=True)
    for candidate in (new,observed):
        assert torch.equal(old[0]['total'],candidate[0]['total'])
        assert torch.equal(old[1],candidate[1])
        assert torch.equal(old[2],candidate[2])
        assert torch.equal(old[3],candidate[3])
    forced=run(MatchedX0Planner.loss,fraction=1.,diagnostics=True)
    assert bool((forced[3]==99).all())
    pairs=forced[0]['diagnostics'];assert pairs.shape==(len(LOSS_DIAGNOSTIC_KEYS),2)
    at=LOSS_DIAGNOSTIC_KEYS.index('all_layers_t99')
    assert torch.allclose(pairs[at,0]/pairs[at,1],forced[0]['x0'],atol=1e-7,rtol=1e-6)
    assert not pairs.requires_grad
    model=MatchedDeformTemporalModel(enable_world=False)
    legacy,legacy_manifest=optimizer_groups(model)
    grouped,manifest=optimizer_groups(model,'differential')
    assert [g['lr'] for g in legacy]==[5e-5,1e-4]
    assert {g['name']:g['lr'] for g in grouped}==dict(image_backbone=5e-6,pretrained_other=1e-5,
        pretrained_planner=2e-5,new_future=1e-4,new_offset=1e-4)
    assert all('.future_' not in name for group in manifest if group['name']=='pretrained_planner' for name in group['parameter_names'])
    assert sum(g['parameter_count'] for g in manifest)==sum(p.numel() for p in model.parameters() if p.requires_grad)
    lowered,lower_manifest=optimizer_groups(model,'differential',offset_lr=1e-5)
    assert next(g['lr'] for g in lowered if g['name']=='new_offset')==1e-5
    first=torch.nn.Parameter(torch.tensor([.1,.2],device='cuda'))
    one=torch.optim.AdamW([first],lr=1e-4,fused=True)
    first.grad=torch.ones_like(first);one.step()
    state=one.state_dict()
    state['state']={key:{name:value.cpu() if torch.is_tensor(value) else value for name,value in values.items()} for key,values in state['state'].items()}
    second=torch.nn.Parameter(first.detach().clone());two=torch.optim.AdamW([second],lr=1e-4,fused=True)
    two.load_state_dict(state);restored=restore_fused_optimizer_devices(two)
    assert all(value.device==parameter.device for parameter,values in two.state.items() for value in values.values() if torch.is_tensor(value))
    first.grad=torch.ones_like(first);second.grad=torch.ones_like(second);one.step();two.step()
    assert torch.equal(first,second),'Fused optimizer resume changed the next update'
    write(output,dict(status='PASS',legacy_loss_equal=True,legacy_gradient_equal=True,
        diagnostic_rng_unchanged=True,t99_option_checked=True,optimizer_partition_checked=True,
        optimizer_groups=manifest,fused_optimizer_resume_checked=True,moved_fused_state_tensors=restored))
    del model;gc.collect()
    print('POLICY_CHECKS_PASS',flush=True)


def world_checks(checkpoint, output):
    from temporal_dataset import TemporalNuScenesDataset,temporal_collate,to_device
    from temporal_matched_deform_model import MatchedDeformTemporalModel,load_matched_deform_state
    from data_paths import DEFAULT_DATA_ROOT
    from benchmark_h20_training import sequence
    from train_h20 import configure_world_batch
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    dataset=TemporalNuScenesDataset(DEFAULT_DATA_ROOT,str(ROOT/'ENV/h20_benchmark_subset.pkl'))
    frames=sequence(dataset)
    model=MatchedDeformTemporalModel(fusion_config={'future_dropout':0.},sampler_config={
        'points_per_map':[36,25,16,9,9],'window_m':[6.,3.],'offset_mode':'learned'}).cuda().eval()
    saved=torch.load(checkpoint,map_location='cpu')
    load_matched_deform_state(model,saved['trainable'],saved['metadata']['architecture_version']);del saved
    def snapshot():
        return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),
            python=random.getstate(),buffers={n:b.detach().clone() for n,b in model.world.named_buffers()},
            modes={n:m.training for n,m in model.world.named_modules()})
    def restore(state):
        torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])
        np.random.set_state(state['numpy']);random.setstate(state['python'])
        with torch.no_grad():
            for n,b in model.world.named_buffers():b.copy_(state['buffers'][n])
        for n,m in model.world.named_modules():m.training=state['modes'][n]
    def equivalent_state(a,b):
        x,y=a['numpy'],b['numpy']
        return (torch.equal(a['cpu'],b['cpu']) and torch.equal(a['cuda'],b['cuda'])
            and x[0]==y[0] and np.array_equal(x[1],y[1]) and x[2:]==y[2:]
            and a['python']==b['python'] and a['modes']==b['modes']
            and all(torch.equal(a['buffers'][n],b['buffers'][n]) for n in a['buffers']))
    results=[]
    for batch_size in (1,2):
        configure_world_batch(model,batch_size);model.reset();captured={}
        def capture(module,inputs):captured['inputs']=inputs
        hook=model.world.register_forward_pre_hook(capture)
        with torch.no_grad():
            for sample in frames[:4]:
                batch=to_device(temporal_collate([copy.deepcopy(sample) for _ in range(batch_size)]),torch.device('cuda'))
                for lane,meta in enumerate(batch['meta']):
                    meta['scene_token']=str(meta['scene_token'])+':equivalence'+str(lane)
                model(batch,compute_loss=False)
        hook.remove();assert 'inputs' in captured
        inputs=captured['inputs'];outputs={};times={};initial=snapshot();after={}
        with torch.no_grad():
            for readout in (6,5):
                restore(initial)
                model.world.readout_steps=readout
                torch.cuda.synchronize();start=time.perf_counter()
                outputs[readout]=model.world(*inputs)
                torch.cuda.synchronize();times[readout]=time.perf_counter()-start
                after[readout]=snapshot()
            assert equivalent_state(after[6],after[5]),'Readout optimization changed RNG or frozen-model state'
            assert torch.equal(outputs[6][:,:5],outputs[5]),'retained world maps changed'
            ego,agents,dense,status,_,canbus=model.base.conditions(batch['camera'],batch['lidar'],batch['status'],batch['command'])
        rotation=batch['lidar_to_ego'][:,:2,:2].float();prior=inputs[3]
        prior_ego=torch.einsum('bij,bkj->bki',rotation,prior)
        forward=torch.linalg.solve(rotation.double(),rotation.new_tensor([1.,0.]).double()[None,:,None].expand(batch_size,-1,-1)).squeeze(-1).float()
        checks=[]
        for precision in ('fp32','bf16'):
            previous=None
            maximum_gradient_error=0.
            for readout in (6,5):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(966);torch.cuda.manual_seed(966)
                with torch.cuda.amp.autocast(enabled=precision=='bf16',dtype=torch.bfloat16):
                    tokens=model.future_encoder(outputs[readout],prior,prior_ego,ego,forward)
                    tokens.update(map_ids=model.future_encoder.map_ids,waypoint_ids=model.future_encoder.waypoint_ids)
                    loss=model.base.planner.loss(batch['trajectory_delta'],batch['mask'],ego,agents,dense,status,
                        tokens,torch.ones(batch_size,device='cuda',dtype=torch.bool),prior_ego.cumsum(1),canbus=canbus)
                loss.backward()
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()
                    if p.grad is not None and (n.startswith('base.planner.') or n.startswith('future_encoder.'))}
                current=(tokens['tokens'].detach().cpu(),loss.detach().cpu(),gradients)
                if previous is not None:
                    assert torch.equal(previous[0],current[0]),'sampled future tokens changed'
                    assert torch.equal(previous[1],current[1]),'planning loss changed'
                    assert previous[2].keys()==current[2].keys()
                    maximum_gradient_error=max(float((previous[2][n]-current[2][n]).abs().max()) for n in gradients)
                    assert all(torch.allclose(previous[2][n],current[2][n],atol=1e-7,rtol=1e-5) for n in gradients),'planner/sampler gradients changed beyond CUDA roundoff tolerance'
                previous=current
            checks.append(dict(precision=precision,tokens_equal=True,loss_equal=True,gradients_allclose=True,
                               maximum_gradient_error=maximum_gradient_error,gradient_atol=1e-7,gradient_rtol=1e-5))
        results.append(dict(batch_size=batch_size,retained_maps_equal=True,downstream=checks,
                            rng_and_buffers_equal=True,
                            world_seconds={'six':times[6],'five':times[5]}))
        print('READOUT_EQUIVALENCE',json.dumps(results[-1]),flush=True)
        del outputs,inputs,captured,previous,gradients,current;gc.collect();torch.cuda.empty_cache()
    write(output,dict(status='PASS',checkpoint=checkpoint,results=results,readout_for_training=5,world_rng_equivalent=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['policy','world'],required=True)
    p.add_argument('--baseline-source');p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode=='policy':policy_checks(args.baseline_source,args.output)
    else:world_checks(args.checkpoint,args.output)


if __name__=='__main__':main()
