"""Measure two task gradients on identical shared parameters without updates/DDP."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_paths import DEFAULT_DATA_ROOT
from temporal_dataset import TemporalNuScenesDataset,temporal_collate,to_device
from temporal_matched_deform_model import MatchedDeformTemporalModel,load_matched_deform_state
from benchmark_h20_training import sequence
from train_h20 import configure_world_batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.cuda.set_device(0);torch.set_num_threads(4);torch.manual_seed(2026)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    data=TemporalNuScenesDataset(DEFAULT_DATA_ROOT,str(ROOT/'ENV/h20_benchmark_subset.pkl'));frames=sequence(data)
    model=MatchedDeformTemporalModel(fusion_config={'future_dropout':0.},sampler_config={
        'points_per_map':[36,25,16,9,9],'window_m':[6.,3.],'offset_mode':'learned'}).cuda().train()
    saved=torch.load(args.checkpoint,map_location='cpu');load_matched_deform_state(model,saved['trainable'],saved['metadata']['architecture_version']);del saved
    configure_world_batch(model,1)
    shared=[p for n,p in model.named_parameters() if n.startswith('base.backbone.') and p.requires_grad]
    records=[]
    for sample in frames:
        batch=to_device(temporal_collate([sample]),torch.device('cuda'))
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):out=model(batch)
        if not bool(out['ready'].all()):continue
        planning=torch.autograd.grad(out['loss_x0'],shared,retain_graph=True,allow_unused=True)
        perception=torch.autograd.grad(out['loss_auxiliary'],shared,allow_unused=True)
        pn=sum((g.float().square().sum() for g in planning if g is not None),torch.zeros((),device='cuda'))
        an=sum((g.float().square().sum() for g in perception if g is not None),torch.zeros((),device='cuda'))
        dot=sum(((a.float()*b.float()).sum() for a,b in zip(planning,perception) if a is not None and b is not None),torch.zeros((),device='cuda'))
        row=dict(token=batch['meta'][0]['token'],planning_norm=float(pn.sqrt()),perception_norm=float(an.sqrt()),
            cosine=float(dot/(pn*an).sqrt().clamp_min(1e-12)),future_used=bool(out['future_used'].all()))
        assert all(torch.isfinite(torch.tensor(row[k])) for k in ('planning_norm','perception_norm','cosine'))
        records.append(row);print('TASK_GRADIENT',json.dumps(row),flush=True)
        del planning,perception,out
    assert records
    payload=dict(status='PASS',shared_parameter_scope='base.backbone.*',checkpoint=args.checkpoint,
        optimizer_updates=0,records=records,limitation='One short real scene and sampled diffusion times; diagnostic only, not a validation estimate.')
    Path(args.output).write_text(json.dumps(payload,indent=2))


if __name__=='__main__':main()
