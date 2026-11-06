# CVCI2_Res

当前架构为 **v6：CVCI2 gate12 x0 规划器 + 冻结世界模型滚动未来条件**。

- 12 层（6 self + 6 interaction），agent → ego → 当前 BEV，保留 Res 未来残差。
- CAN BUS 原条件路径及轨迹 token 拼接；CAN BUS/导航独立 SE 当前 BEV 门控。
- 12 层 x0 MSE 等权平均，检测/框回归/occupancy 联合监督。
- 以本机 Models 中 CVCI2 最佳权重初始化，感知、规划器与未来编码器全部联合训练。
- IRWM 保持冻结；三帧历史、六步 rollout 和五个同时间未来图采样规则保留。

详细迁移、检查点和本机使用方法见 [GATE12_MIGRATION.md](docs/GATE12_MIGRATION.md)。
环境说明见 [ENV_SETUP.md](docs/ENV_SETUP.md)。

## 数据与权重

默认数据入口为 `/mnt/workspace/dataSet/nuscene`，可通过 `--data-root` 或
`NUSCENES_DATA_ROOT` 覆盖。需要带轨迹、掩码、指令及 ego_status 的
`nuscenes_infos_train.pkl` 和 `nuscenes_infos_val.pkl`。
默认初始化权重为 `/mnt/workspace/nix3sgh/Models/best_endpoint_1step.pth`；
世界模型权重为 `/mnt/workspace/nix3sgh/Models/MMO_MSO_with_plan_tightly_coupled.pth`。
实际 IRWM 还需通过 `--world-root` 提供 Drive-OccWorld 配置。

## 入口

```bash
cd /mnt/workspace/nix3sgh/CVCI2_Res
source ENV/bin/activate
python tools/train_temporal_matched_deform --help
python tools/eval_cvci3_protocol.py --help
python tools/validate_matched_deform --output ENV/gate12_full_report.json
```

v6 检查点含完整联合训练状态及感知 BN buffers。旧 v5 epsilon 检查点需配合
迁移前备份源码使用。`docs/` 中标注 V5 的文档是历史记录；Slurm 脚本仍使用原 HPC
调度和路径设置，当前本机直接使用上述 Python 入口。
`transfuser_dit.py` 与 `train_nusc.py` 保留为历史 baseline 工具；当前 Res 入口使用
`cvci2_gate12/`、`matched_diffusion.py` 和 `temporal_matched_deform_model.py`。
