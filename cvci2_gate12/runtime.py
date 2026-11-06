"""Compatibility for the project-local Torch 2.0 runtime on Hopper GPUs."""
import torch


def configure_attention_backend():
    version=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if version <= (2,0) and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        # Old fused SDP kernels are unreliable on this H20 runtime, including backward.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        return 'math'
    return 'automatic'
