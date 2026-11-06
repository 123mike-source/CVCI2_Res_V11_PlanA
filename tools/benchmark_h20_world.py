from pathlib import Path
import json,math,time,sys,traceback
import numpy as np
import torch
r=Path('/mnt/workspace/nix3sgh/CVCI2_Res');sys.path.insert(0,str(r));sys.path.insert(0,str(r/'tools'))
from temporal_matched_deform_model import MatchedDeformTemporalModel
from validate_gate12_migration import synthetic_batch
torch.set_num_threads(4);torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
report={'phase':'loading_real_irwm','status':'RUNNING'}
out=r/'ENV/h20_real_world_smoke.json'
def save():out.write_text(json.dumps(report,indent=2))
save()
try:
 t=time.perf_counter();model=MatchedDeformTemporalModel(fusion_config={'future_dropout':0.},sampler_config={'points_per_map':[36,25,16,9,9],'window_m':[6.,3.],'offset_mode':'learned'}).cuda().eval()
 report.update(phase='forward',load_seconds=time.perf_counter()-t,world_parameters=sum(p.numel() for p in model.world.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad));save()
 for f in range(4):
  batch=synthetic_batch(f,'cuda');batch['camera']=torch.randn(1,3,256,1024,device='cuda');batch['world_image']=torch.randn(1,6,3,256,704,device='cuda')
  projections=[]
  intrinsic=np.array([[400.,0,352.],[0,400.,128.],[0,0,1.]])
  for c in range(6):
   a=c*math.pi/3;rot=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
   camera=np.array([[0,-1.,0],[0,0,-1.],[1.,0,0]])@rot
   p=np.eye(4);p[:3,:3]=intrinsic@camera;p[:3,3]=intrinsic@np.array([0,1.5,0]);projections.append(p)
  batch['projection']=torch.tensor(np.stack(projections),device='cuda',dtype=torch.float32)[None]
  torch.cuda.synchronize();t=time.perf_counter()
  if f==3:
   model.train()
   with torch.cuda.amp.autocast(dtype=torch.bfloat16):result=model(batch,profile=True,ddim_steps=1)
   result['loss'].backward()
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
  else:
   with torch.no_grad(),torch.cuda.amp.autocast(dtype=torch.bfloat16):result=model(batch,profile=True,ddim_steps=1)
  torch.cuda.synchronize();seconds=time.perf_counter()-t
  print('REAL_WORLD_FRAME',f,'seconds',seconds,'future_used',result['future_used'].tolist(),'timings',result['timings'],flush=True)
  if f==3:
   report.update(status='PASS',phase='complete',active_frame_seconds=seconds,timings=result['timings'],loss=float(result['loss']),grad_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
   assert bool(result['future_used'].all())
   assert torch.isfinite(result['prediction']).all()
 save();print('H20_REAL_IRWM_SMOKE_PASS',json.dumps(report),flush=True)
except BaseException as exc:
 report.update(status='FAIL',phase='failed',error=repr(exc));save();traceback.print_exc();raise
