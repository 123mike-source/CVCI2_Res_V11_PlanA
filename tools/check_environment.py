#!/usr/bin/env python
"""Verify the project-local runtime without datasets or pretrained checkpoints."""
import argparse,importlib,importlib.metadata as metadata,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',default=str(ROOT/'ENV/environment_report.json'));args=p.parse_args()
report={'status':'RUNNING','python':sys.version,'executable':sys.executable,'checks':{}}
def passed(name,details=True):
    report['checks'][name]=details;print('PASS',name,details,flush=True)
try:
    assert sys.prefix==str(ROOT/'ENV'),'Activate CVCI2_Res/ENV first'
    for name in ['torch','torchvision','mmcv','mmdet','timm','numpy','scipy','cv2','nuscenes','pyquaternion','tensorboard','numba']:
        importlib.import_module(name)
    passed('core_imports')
    report['versions']={name:metadata.version(name) for name in ['torch','torchvision','mmcv-full','mmdet','timm','numpy','scipy','opencv-python','nuscenes-devkit','tensorboard']}
    import world_models.irwm_wrapper
    import irwm_projects.mmdet3d_plugin
    passed('irwm_runtime_imports')
    import torch
    torch.set_num_threads(2)
    if args.device=='cuda':
        assert torch.cuda.is_available(),'CUDA is unavailable'
        passed('cuda_devices',{'count':torch.cuda.device_count(),'selected':torch.cuda.get_device_name(0),'runtime':torch.version.cuda})
    device=torch.device(args.device)
    x=torch.randn(16,16,device=device,requires_grad=True);(x@x.T).square().mean().backward();assert torch.isfinite(x.grad).all();passed('forward_backward')
    from torchvision.ops import nms
    boxes=torch.tensor([[0.,0.,2.,2.],[0.,0.,1.,1.],[3.,3.,4.,4.]],device=device)
    keep=nms(boxes,torch.tensor([.9,.8,.7],device=device),.5);assert keep.numel()==3;passed('torchvision_nms')
    if args.device=='cuda':
        assert torch.cuda.is_bf16_supported(),'BF16 is unavailable'
        layer=torch.nn.Conv2d(3,8,3,padding=1).to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):y=layer(torch.randn(2,3,16,16,device=device))
        assert y.dtype==torch.bfloat16;y.float().square().mean().backward();assert torch.isfinite(layer.weight.grad).all();passed('bf16_convolution_backward')
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction,multi_scale_deformable_attn_pytorch
        value=torch.randn(1,4,1,8,device=device,requires_grad=True)
        shapes=torch.tensor([[2,2]],dtype=torch.long,device=device);starts=torch.tensor([0],dtype=torch.long,device=device)
        locations=torch.rand(1,2,1,1,2,2,device=device,requires_grad=True)
        weights=torch.full((1,2,1,1,2),.5,device=device,requires_grad=True)
        actual=MultiScaleDeformableAttnFunction.apply(value,shapes,starts,locations,weights,64)
        expected=multi_scale_deformable_attn_pytorch(value,shapes,locations,weights)
        assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-4)
        actual.square().sum().backward();assert torch.isfinite(value.grad).all();passed('mmcv_cuda_deformable_attention_backward')
    from world_models.matched_prior_sampler import MatchedPriorSampler
    sampler=MatchedPriorSampler(dict(points_per_map=[36,25,16,9,9],window_m=[6.,3.],offset_mode='learned')).to(device)
    bev=torch.randn(1,6,256,200,200,device=device)
    prior=torch.tensor([[[.4,2.]]],device=device).expand(1,6,2)
    result=sampler(bev,prior,prior,torch.randn(1,1,256,device=device),torch.tensor([[1.,0.]],device=device))
    assert result['tokens'].shape==(1,95,256)
    result['tokens'].square().mean().backward()
    assert sampler.offset.weight.grad is not None and torch.isfinite(sampler.offset.weight.grad).all()
    passed('project_95_token_sampler_backward',list(result['tokens'].shape))
    del bev,sampler,result
    from transfuser_dit import TransFuserBackbone
    backbone=TransFuserBackbone().to(device).train()
    image=torch.randn(1,3,64,256,device=device);lidar=torch.randn(1,1,64,64,device=device)
    if args.device=='cuda':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):features,dense=backbone(image,lidar)
    else:features,dense=backbone(image,lidar)
    assert torch.isfinite(dense).all();dense.float().square().mean().backward()
    passed('project_transfuser_forward_backward',list(dense.shape))
    report['status']='PASS'
except BaseException as exc:
    report['status']='FAIL';report['error']=repr(exc);traceback.print_exc()
finally:
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if report['status']!='PASS':raise SystemExit(1)