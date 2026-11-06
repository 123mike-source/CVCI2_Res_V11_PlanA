"""Current-BEV supervision with explicit metric layout and valid masks."""
import torch
import torch.nn.functional as F


def sparse_depth_loss(logits_by_scale, batch):
    terms, counts = [], []
    for logits, stride in zip(logits_by_scale, (16, 8)):
        target = batch['geobev_depth_s%d' % stride].float().flatten(0, 1)
        if logits.shape[-2:] != target.shape[-2:] or logits.shape[1] != 119:
            raise ValueError('Depth loss scale/channel contract differs')
        valid = torch.isfinite(target) & (target >= 1) & (target < 60)
        bins = ((target - 1) / .5).floor().long().clamp(0, 117)
        # Channel 118 is a pretrained foreground classifier, never a depth bin.
        error = F.cross_entropy(logits[:, :118].float(), bins, reduction='none')
        b = len(batch['camera'])
        numerator = (error * valid).reshape(b, -1).sum(-1)
        denominator = valid.reshape(b, -1).sum(-1)
        terms.append((numerator / denominator.clamp_min(1)).mean())
        counts.append(denominator.sum().detach())
    return torch.stack(terms).mean(), torch.stack(counts).sum()


def masked_occupancy_loss(logits, lidar_target, visible):
    # Histogram and planner features both have row=x_ego, col=y_ego, range ±32m.
    if logits.shape != visible.shape or logits.shape[-2:] != (64, 64):
        raise ValueError('Occupancy feature/visibility layout differs')
    target = (F.max_pool2d(lidar_target.float(), 4) > 0).float()
    if target.shape != logits.shape:
        raise ValueError('Occupancy targets must be metric-aligned 256→64 max-pooling')
    valid = visible.float()
    positive = (target * valid).sum((1, 2, 3))
    count = valid.sum((1, 2, 3))
    positive_weight = ((count - positive) / positive.clamp_min(1)).clamp(1, 20)[:, None, None, None]
    error = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none',
                                               pos_weight=positive_weight)
    bce = (error * valid).sum((1, 2, 3)) / count.clamp_min(1)
    prob = logits.float().sigmoid() * valid
    masked_target = target * valid
    dice = 1 - (2 * (prob * masked_target).sum((1, 2, 3)) + 1) / (
        prob.sum((1, 2, 3)) + masked_target.sum((1, 2, 3)) + 1)
    return (.5 * (bce + dice) * (count > 0)).mean(), count.sum().detach()
