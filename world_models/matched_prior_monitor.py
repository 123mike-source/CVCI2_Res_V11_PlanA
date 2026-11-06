"""Low-frequency, denominator-weighted DDP monitoring. Does not draw RNG."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from temporal_matched_deform_model import monitoring_keys


def gradient_monitor(model):
    result = {}
    groups = dict(dit='base.planner.', offset='future_encoder.offset.', pointwise_encoder='future_encoder.value_projection.',
                  perception='base.backbone.', spatial_gate='base.spatial_gate.')
    if hasattr(model, 'geobev_config'):
        groups['perception'] = 'base.geobev.'
        groups['bev_bridge'] = 'base.bridge.'
    device = next(model.parameters()).device
    for key, prefix in groups.items():
        norms = [p.grad.detach().float().norm() for n,p in model.named_parameters()
                 if n.startswith(prefix) and p.grad is not None]
        norm = torch.stack(norms).norm() if norms else torch.zeros((), device=device)
        result[f'gradient/{key}_norm'] = torch.stack((norm, torch.ones_like(norm)))
    return result


def write_monitor(model, extra, writer, work_dir, epoch, micro_step, distributed):
    # Extra optimizer/control keys are emitted with the same schema on every rank.
    keys = sorted(set(monitoring_keys()) | set(extra))
    device = next(model.parameters()).device
    stats = {**model.last_monitor, **extra}
    zero = torch.zeros(2, device=device)
    packed = torch.stack([stats.get(k, zero) for k in keys]).double()
    if distributed:
        dist.all_reduce(packed)
    if writer is None:
        return
    values = packed.cpu().numpy()
    metrics = {k:float(values[i,0]/values[i,1]) for i,k in enumerate(keys) if values[i,1]>0}
    if not all(np.isfinite(x) for x in metrics.values()):
        raise FloatingPointError('non-finite monitoring scalar')
    for key, value in metrics.items():
        writer.add_scalar('monitor/'+key, value, micro_step)
    matrix_keys = [f'attention/reference_bev/j{j+1}_h{h+1}' for j in range(6) for h in range(6)]
    if all(k in metrics for k in matrix_keys):
        matrix = np.array([metrics[k] for k in matrix_keys], np.float32).reshape(6,6)
        image = np.kron(matrix/max(float(matrix.max()), 1e-9), np.ones((24,24),np.float32))
        writer.add_image('monitor/attention/reference_bev_heatmap', image, micro_step, dataformats='HW')
    event = dict(event='monitor', epoch=epoch, micro_step=micro_step,
                 scope='global_monitored_microbatch_weighted_by_valid_samples', metrics=metrics)
    with open(Path(work_dir)/'monitoring.jsonl', 'a') as stream:
        stream.write(json.dumps(event)+'\n')
    summary = dict(event='monitor', epoch=epoch, micro_step=micro_step,
                   valid_fraction=[metrics.get(f'sampler/valid_fraction/h{h+1}') for h in range(5)],
                   attention_mass=[metrics.get(f'attention/mass_bev/h{h+1}') for h in range(5)],
                   train_batch_ddim_l2_m=[metrics.get(f'planning/train_batch_ddim_l2_m/{s}s') for s in (1,2,3)],
                   outside_window_fraction=metrics.get('planning/outside_window_fraction'),
                   retained_prior_l2_m=metrics.get('planning/retained_prior_l2_m'),
                   extrapolated_prior_l2_m=metrics.get('planning/extrapolated_prior_l2_m'),
                   offset_grad_norm=metrics.get('gradient/offset_norm'),
                   encoder_seconds=metrics.get('timing/future_encoder_seconds'))
    print(json.dumps(summary), flush=True)
    writer.flush()
