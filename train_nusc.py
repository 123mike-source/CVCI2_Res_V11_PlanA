"""Four-GPU nuScenes training for the language-free TransFuser DiT baseline."""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from data_paths import DEFAULT_DATA_ROOT
from nuscenes_dataset import NuScenesTransFuserDataset
from transfuser_dit import NuScenesTransFuserDiT


def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = torch.zeros(7, device=device)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(10000 + dist.get_rank())
        for batch in loader:
            batch = to_device(batch, device)
            with autocast(dtype=torch.bfloat16):
                prediction = model.sample(
                    batch["camera"], batch["lidar"], batch["status"], batch["command"]
                )
            target = batch["trajectory_delta"].cumsum(dim=1)
            l2 = (prediction.float() - target).square().sum(dim=-1).sqrt()
            mask = batch["mask"]
            for slot, point in enumerate((1, 3, 5)):
                totals[slot] += (l2[:, point] * mask[:, point]).sum()
                totals[3 + slot] += mask[:, point].sum()
            totals[6] += batch["camera"].shape[0]
    dist.all_reduce(totals)
    return [float((totals[i] / totals[i + 3].clamp_min(1)).item()) for i in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="nuScenes data root (default: %(default)s)")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    local_rank, rank, world, device = setup()
    torch.manual_seed(2026 + rank)
    np.random.seed(2026 + rank)
    random.seed(2026 + rank)

    train_data = NuScenesTransFuserDataset(args.data_root, os.path.join(args.data_root, "nuscenes_infos_train.pkl"))
    val_data = NuScenesTransFuserDataset(args.data_root, os.path.join(args.data_root, "nuscenes_infos_val.pkl"))
    train_sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=2026)
    val_sampler = DistributedSampler(val_data, world, rank, shuffle=False)
    train_loader = DataLoader(train_data, args.batch_size, sampler=train_sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_data, args.batch_size, sampler=val_sampler, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)

    base = NuScenesTransFuserDiT(args.image_checkpoint).to(device)
    model = DDP(base, device_ids=[local_rank], find_unused_parameters=False)
    image_params, other_params = [], []
    for name, parameter in model.module.named_parameters():
        (image_params if name.startswith("backbone.image.") else other_params).append(parameter)
    optimizer = AdamW([{"params": image_params, "lr": args.lr * 0.5},
                       {"params": other_params, "lr": args.lr}], weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)

    def schedule(step):
        if step < 500:
            return (step + 1) / 500.0
        progress = (step - 500) / max(total_steps - 500, 1)
        return 1e-3 + 0.5 * (1 - 1e-3) * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, schedule)
    start_epoch, best_l2 = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_l2 = state["epoch"] + 1, state["best_l2"]
    os.makedirs(args.work_dir, exist_ok=True)
    if rank == 0:
        print("training world=%d per_gpu_batch=%d global_batch=%d steps_per_epoch=%d epochs=%d" %
              (world, args.batch_size, world * args.batch_size, len(train_loader), args.epochs), flush=True)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running = torch.zeros(6, device=device)
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                loss, components = model(
                    batch["camera"], batch["lidar"], batch["status"], batch["command"],
                    batch["trajectory_delta"], batch["mask"],
                    batch["agent_states"], batch["agent_labels"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += torch.cat((loss.detach().view(1), components, loss.new_ones(1)))
            if rank == 0 and (step + 1) % 20 == 0:
                values = running[:5] / running[5]
                print(
                    "epoch=%d step=%d/%d total=%.6f diffusion=%.6f "
                    "agent_cls=%.6f agent_box=%.6f occupancy=%.6f lr=%.2e" %
                    (epoch + 1, step + 1, len(train_loader), *values,
                     scheduler.get_last_lr()[1]), flush=True
                )
            if args.max_steps and step + 1 >= args.max_steps:
                break
        dist.all_reduce(running)
        l2_1s, l2_2s, l2_3s = validate(model.module, val_loader, device)
        mean_l2 = (l2_1s + l2_2s + l2_3s) / 3
        if rank == 0:
            state = {"epoch": epoch, "model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "best_l2": min(best_l2, mean_l2),
                     "l2_1s": l2_1s, "l2_2s": l2_2s, "l2_3s": l2_3s}
            torch.save(state, os.path.join(args.work_dir, "last.pth"))
            if mean_l2 < best_l2:
                best_l2 = mean_l2
                torch.save(state, os.path.join(args.work_dir, "best_l2.pth"))
            values = running[:5] / running[5]
            print(
                "epoch=%d train_total=%.6f diffusion=%.6f agent_cls=%.6f "
                "agent_box=%.6f occupancy=%.6f L2@1s=%.4f L2@2s=%.4f "
                "L2@3s=%.4f best_mean=%.4f" %
                (epoch + 1, *values, l2_1s, l2_2s, l2_3s, best_l2), flush=True
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
