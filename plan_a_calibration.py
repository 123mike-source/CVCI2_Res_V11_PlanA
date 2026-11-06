"""Training-only position-loss calibration; never performs optimizer updates."""
import hashlib
import json
from pathlib import Path
import statistics

import torch
import torch.distributed as dist


def run(model, loader, sampler, work, batches=12, precision='bf16'):
    if not 8 <= batches <= 16:
        raise ValueError('Calibration uses 8..16 training batches')
    if model.plan_a_config['position_loss_weight'] != 0:
        raise ValueError('Calibration must start with coefficient zero')
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    device = next(model.parameters()).device
    if distributed:
        # A real DDP constructor broadcasts parameters/buffers. This calibration
        # uses autograd.grad without DDP hooks, so perform that broadcast here.
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    named = [(n, p) for n, p in model.base.planner.named_parameters()
             if p.requires_grad and n.startswith(('head.', 'layers.9.', 'layers.10.'))]
    parameters = [p for _, p in named]
    model.train()
    model.base.planner.position_loss_update = 0
    initial_hash = _hash_state(model)
    rows, tokens = [], []
    iterator = iter(loader)
    from temporal_dataset import to_device
    for step in range(batches):
        batch = to_device(next(iterator), device)
        tokens.append([m['token'] for m in batch['meta']])
        with torch.autocast(device_type=device.type, enabled=precision == 'bf16', dtype=torch.bfloat16):
            output = model(batch, monitor=False)
        gradients = []
        for index, key in enumerate(('loss_x0', 'loss_position')):
            values = torch.autograd.grad(output[key], parameters, retain_graph=index == 0, allow_unused=True)
            # All-reduce the gradient vectors FIRST, then take a norm. This is
            # the norm of the global batch gradient, not a mean of rank norms.
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                                for p, g in zip(parameters, values)])
            if distributed:
                dist.all_reduce(vector)
                vector.div_(world)
            gradients.append(float(vector.norm()))
        sampler.commit()
        row = dict(batch=step+1, x0_gradient_norm=gradients[0], position_gradient_norm=gradients[1],
            local_x0=float(output['loss_x0']), local_position_m=float(output['loss_position']),
            local_ready=int(output['ready'].sum()), local_future_used=int(output['future_used'].sum()))
        rows.append(row)
        if rank == 0:
            print(json.dumps(dict(event='position_calibration', **row)), flush=True)
        del output
    x0 = statistics.median(r['x0_gradient_norm'] for r in rows)
    position = statistics.median(r['position_gradient_norm'] for r in rows)
    if not x0 > 0 or not position > 0:
        raise FloatingPointError('Calibration gradients are zero/nonfinite')
    coefficient = .2*x0/position
    if not 0 < coefficient < 100:
        raise FloatingPointError('Position calibration coefficient is invalid')
    hashes = [None]*world
    if distributed:
        dist.all_gather_object(hashes, dict(rank=rank, tokens=tokens))
    else:
        hashes[0] = dict(rank=rank, tokens=tokens)
    result = dict(status='PASS', schema='plan_a_global_gradient_calibration_v1',
        split='nuscenes_infos_train.pkl_only', batches=batches, world_size=world,
        target_gradient_ratio=.2, x0_median_norm=x0, position_median_norm=position,
        position_loss_weight=coefficient, ramp_updates=500, rows=rows,
        parameter_names=[n for n, _ in named], initial_parameter_sha256=initial_hash,
        final_parameter_sha256=_hash_state(model), optimizer_updates=0,
        streams_by_rank=hashes, notes='Discard calibration instance; reconstruct fresh model/optimizer/RNG/rolling for training')
    assert result['initial_parameter_sha256'] == result['final_parameter_sha256']
    # Parameters stay fixed; BN buffers intentionally follow train mode. They
    # belong only to this disposable calibration model and are never reused.
    if rank == 0:
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        (work/'position_calibration.json').write_text(json.dumps(result, indent=2))
        print('POSITION_CALIBRATION_RESULT '+json.dumps({k: result[k] for k in
              ('status', 'position_loss_weight', 'x0_median_norm', 'position_median_norm', 'optimizer_updates')}), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return result


def _hash_state(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
