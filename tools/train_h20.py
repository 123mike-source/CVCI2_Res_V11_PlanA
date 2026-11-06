"""H20 execution settings for the existing v6 trainer; model math is unchanged."""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def configure_world_batch(model,batch_size):
    # MMCV requires the effective batch to divide min(batch, im2col_step).
    # Six-camera batches can exceed the old limit of 64. Process the full
    # per-frame camera batch; this changes kernel partitioning only.
    step=max(64,6*batch_size)
    count=0
    for module in model.world.modules():
        if hasattr(module,'im2col_step'):
            module.im2col_step=step;count+=1
    return dict(im2col_step=step,modules=count)


def configure_loader_worker(worker_id):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def main():
    if '--bev-encoder' in sys.argv and sys.argv[sys.argv.index('--bev-encoder')+1] == 'geobev':
        from tools.train_geobev_h20 import main as geobev_main
        return geobev_main()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    deterministic_checks = os.environ.get('CVCI_DETERMINISTIC_CHECKS') == '1'
    if deterministic_checks:
        # Limit this to the short restoration regression; production retains
        # H20 autotuning. cuDNN can otherwise choose different convolution
        # kernels in two independent processes, before any interruption.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    import cv2
    cv2.setNumThreads(1)
    namespace=runpy.run_path(str(ROOT/'tools/train_temporal_matched_deform'),run_name='h20_trainer')
    entry=namespace['main'];globals_=entry.__globals__
    loader=globals_['DataLoader'];adamw=globals_['AdamW'];ddp=globals_['DDP']
    base_model=globals_['JointGate12TemporalModel']

    def fast_loader(*args,**kwargs):
        if kwargs.get('num_workers',0)>0:
            kwargs['prefetch_factor']=int(os.environ.get('CVCI_PREFETCH_FACTOR','2'))
            kwargs['worker_init_fn']=configure_loader_worker
        return loader(*args,**kwargs)

    def fused_adamw(*args,**kwargs):
        kwargs['fused']=True
        return adamw(*args,**kwargs)

    def fast_ddp(*args,**kwargs):
        kwargs.update(gradient_as_bucket_view=True,bucket_cap_mb=64)
        return ddp(*args,**kwargs)

    class AuditedModel(base_model):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.world is not None and not self.world.training
            assert all(p.requires_grad == self.is_trainable('world.'+n) for n,p in self.world.named_parameters())
            assert not any(m.training for m in self.world.modules())
            assert all(p.requires_grad for p in self.base.parameters())
            assert all(float(layer.future_gate)==0 for layer in self.base.planner.layers if hasattr(layer,'future_gate'))
            native_batch=configure_world_batch(self,int(sys.argv[sys.argv.index('--batch-size')+1]))
            if int(os.environ.get('RANK','0'))==0:
                from planning_runtime import DEFAULT_BASE
                source=torch.load(args[0] if args and args[0] is not None else DEFAULT_BASE,map_location='cpu')['model']
                current=self.base.state_dict()
                assert all(torch.equal(current[name],value) for name,value in source.items())
                assert all('.future_' in name for name in set(current)-set(source))
                work=Path(sys.argv[sys.argv.index('--work-dir')+1]);work.mkdir(parents=True,exist_ok=True)
                report=dict(stage='construction_before_optional_A9_warm_start',
                    cvci2_shared_initialization='best_checkpoint',shared_tensors_verified=len(source),
                    cvci2_existing_se_gates='pretrained_as_part_of_original_cvci2',
                    res_future_attention='from_scratch',res_bev_sampler='from_scratch',
                    res_future_gates='from_scratch_zero_residual_gate',
                    irwm='pretrained_partial_tuning' if self.world_tuning_config is not None else 'pretrained_frozen',attention_backend=self.base.planner.attention_backend,
                    precision='bf16_cvci2_fp32_tf32_irwm',optimizer='fused_adamw')
                report['deformable_attention_batch']=native_batch
                report['deterministic_validation']=deterministic_checks
                report['cudnn_benchmark']=torch.backends.cudnn.benchmark
                (work/'initialization_audit.json').write_text(json.dumps(report,indent=2))
                print('INITIALIZATION_AUDIT',json.dumps(report),flush=True)

    globals_.update(DataLoader=fast_loader,AdamW=fused_adamw,DDP=fast_ddp,
                    JointGate12TemporalModel=AuditedModel)
    entry()


if __name__=='__main__':main()
