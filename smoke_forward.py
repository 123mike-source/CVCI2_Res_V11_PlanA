"""One-real-sample forward/backward/DDIM smoke test."""

import torch
from torch.cuda.amp import autocast

from data_paths import DEFAULT_DATA_ROOT
from nuscenes_dataset import NuScenesTransFuserDataset
from transfuser_dit import NuScenesTransFuserDiT


root = DEFAULT_DATA_ROOT
dataset = NuScenesTransFuserDataset(root, root + "/nuscenes_infos_train.pkl")
sample = {key: value[None].cuda() for key, value in dataset[0].items()}
model = NuScenesTransFuserDiT(
    "/share/home/u21012/nixiran/CVCI2/ckpts/resnet34-b627a593.pth"
).cuda().train()
with autocast(dtype=torch.bfloat16):
    loss, components = model(
        sample["camera"], sample["lidar"], sample["status"], sample["command"],
        sample["trajectory_delta"], sample["mask"],
        sample["agent_states"], sample["agent_labels"],
    )
loss.backward()
model.eval()
with autocast(dtype=torch.bfloat16):
    prediction = model.sample(
        sample["camera"], sample["lidar"], sample["status"], sample["command"]
    )
print("camera", tuple(sample["camera"].shape), "lidar", tuple(sample["lidar"].shape))
print("status", tuple(sample["status"].shape), "command", tuple(sample["command"].shape))
print("agents", tuple(sample["agent_states"].shape), "positive", int(sample["agent_labels"].sum()))
print("target_delta", tuple(sample["trajectory_delta"].shape), "components", components.float().tolist())
print("loss", float(loss), "prediction", tuple(prediction.shape),
      "finite", bool(prediction.isfinite().all()))
print("max_memory_gib", torch.cuda.max_memory_allocated() / 1024 ** 3)
