"""H20 entry for GeoBEV V9, using the existing serial-scene training protocol."""
import json
import os
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    defaults = {'--bev-encoder': ['geobev'], '--optimizer-profile': ['differential'],
                '--world-readout-steps': ['5'], '--points-per-map': ['9', '9', '16', '25', '36']}
    for key, values in defaults.items():
        if key not in sys.argv:
            sys.argv.extend([key, *values])
    required_flags = ['--integrated-motion-gate', '--adaptive-sampling', '--tune-world']
    fresh_default = '--warm-start-geobev' not in sys.argv
    if '--resume' in sys.argv:
        preview = torch.load(sys.argv[sys.argv.index('--resume') + 1], map_location='cpu')
        fresh_default = bool(preview['metadata']['config'].get('fresh_res', False))
        del preview
    if fresh_default:
        required_flags.append('--fresh-res')
    for key in required_flags:
        if key not in sys.argv:
            sys.argv.append(key)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    checks = os.environ.get('CVCI_DETERMINISTIC_CHECKS') == '1'
    torch.backends.cudnn.benchmark = not checks
    torch.backends.cudnn.deterministic = checks
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    import cv2
    cv2.setNumThreads(1)
    import tools.train_h20 as settings
    from geobev_encoder.model import GeoBEVTemporalModel
    namespace = runpy.run_path(str(ROOT / 'tools/train_temporal_matched_deform'), run_name='geobev_trainer')
    entry = namespace['main']
    globals_ = entry.__globals__
    loader, adam, ddp = globals_['DataLoader'], globals_['AdamW'], globals_['DDP']

    def fast_loader(*args, **kwargs):
        if kwargs.get('num_workers', 0):
            kwargs['prefetch_factor'] = int(os.environ.get('CVCI_PREFETCH_FACTOR', '2'))
            kwargs['worker_init_fn'] = settings.configure_loader_worker
        return loader(*args, **kwargs)

    def fused_optimizer(*args, **kwargs):
        return adam(*args, fused=True, **kwargs)

    def fast_ddp(*args, **kwargs):
        kwargs.update(gradient_as_bucket_view=True, bucket_cap_mb=64)
        return ddp(*args, **kwargs)

    class AuditedGeoBEV(GeoBEVTemporalModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            batch = int(sys.argv[sys.argv.index('--batch-size') + 1]) if '--batch-size' in sys.argv else 6
            settings.configure_world_batch(self, batch)
            assert not any(n.startswith(('backbone.', 'bev_down.', 'dense_proj.')) for n in self.base.state_dict())
            assert self.base.geobev.initialization_audit['status'] == 'PASS'

    globals_.update(DataLoader=fast_loader, AdamW=fused_optimizer, DDP=fast_ddp,
                    JointGate12TemporalModel=AuditedGeoBEV)
    entry()


if __name__ == '__main__':
    main()
