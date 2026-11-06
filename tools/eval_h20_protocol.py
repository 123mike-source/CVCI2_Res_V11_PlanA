"""H20 execution settings around the existing, unchanged serial evaluation protocol."""
import os
from pathlib import Path
import sys

import cv2
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def configure_worker(worker_id):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def main():
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=True
    cv2.setNumThreads(1)
    from eval_cvci3_protocol import main as entry
    globals_=entry.__globals__;loader=globals_['DataLoader']
    def fast_loader(*args,**kwargs):
        if kwargs.get('num_workers',0)>0:kwargs['worker_init_fn']=configure_worker
        return loader(*args,**kwargs)
    globals_['DataLoader']=fast_loader
    readout = int(os.environ.get('CVCI_WORLD_READOUT_STEPS', '6'))
    if readout not in (5, 6):
        raise ValueError('CVCI_WORLD_READOUT_STEPS must be 5 or 6')
    if readout == 5:
        from world_models.irwm_wrapper import FrozenIRWMFutureBEV
        # This readout is validated separately against the original six-map
        # path. The serial sample/noise/mask evaluation protocol is unchanged.
        FrozenIRWMFutureBEV.readout_steps = 5
    entry()


if __name__=='__main__':main()
