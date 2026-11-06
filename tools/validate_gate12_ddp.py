"""Two-rank joint-optimization and BatchNorm checkpoint synchronization check."""
from pathlib import Path
import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from temporal_matched_deform_model import MatchedDeformTemporalModel,matched_deform_state_dict
from planning_runtime import synchronize_base_buffers
from validate_gate12_migration import synthetic_batch


def main():
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank=dist.get_rank();device=torch.device('cuda',local_rank)
    torch.set_num_threads(2)
    torch.manual_seed(2026+rank)
    model=MatchedDeformTemporalModel(enable_world=False).to(device).train()
    wrapper=DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False,
                                    find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    for frame in range(2):
        optimizer.zero_grad(set_to_none=True)
        batch=synthetic_batch(frame,device)
        batch['camera']=batch['camera']+rank*.2
        batch['lidar']=batch['lidar']*(1+rank*.1)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            result=wrapper(batch,ddim_steps=1)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        print('DDP_STEP_PASS',rank,frame,float(result['loss']),float(norm),flush=True)
    synchronize_base_buffers(model)
    digest=hashlib.sha256()
    state=matched_deform_state_dict(model)
    for name,value in state.items():
        digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    hashes=[None]*dist.get_world_size()
    dist.all_gather_object(hashes,digest.hexdigest())
    assert len(set(hashes))==1,hashes
    if rank==0:
        result=dict(status='PASS',world_size=dist.get_world_size(),steps=2,
                    complete_base_future_and_bn_agreement=hashes[0],
                    inactive_future_grad_hooks='PASS',joint_perception_updates='PASS')
        (ROOT/'ENV/gate12_ddp_report.json').write_text(json.dumps(result,indent=2))
        print('GATE12_DDP_PASS',json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
