# CVCI2_Res V11 Plan A — completed fresh 10-epoch training

本仓库保存 **V11 方案 A 从原始预训练权重完整训练 10 轮**所用源码、配置和结果。服务器实验：`h20_v11_plan_a_fresh10_20261105_01`。

已逐文件核对训练记录中的 **74 个 Python 源码 SHA256**，另核对无扩展名的主训练脚本、原始启动配置与第三方运行时清单。全量有效 5119 帧的最佳 MeanPrefix L2 为 **0.336186 m**，对应第 10 轮。

## 1. 网络结构

```text
六相机当前/历史图像 ── GeoBEV ── 当前 BEV、agent / ego 条件
历史观测和轨迹 prior ── 预训练 IRWM ── 一次五步未来滚动、95 tokens
                                          ↓
轨迹噪声 ── S1 ── [门控交互 I2 → self S3] ── … ── [I10 → S11]
              各子层读出 x0，分离梯度的累计位置更新下一层采样
                                          ↓
                                  第 11 层最终轨迹
```

- DiT 256 维、6 个 waypoint token，预测每 0.5 秒的干净增量位移 `x0`，最远 3 秒。
- 采用 **stem S1 + 5×[interaction, self]**：保留原第 1–11 层，删除末尾 I12，共 6 self、5 interaction。所有 11 个子层读出均保留。
- 交互依次处理 agent、ego、当前 BEV，再融合原有可靠性门控的未来条件，之后保留原 FFN 与时间/状态调制。
- 当前 BEV 在候选位置的中心、切向前后、法向左右读取 5 点。未来每次五步滚动一次，按 `[9,9,16,25,36]` 读取 95 tokens，并保留几何距离偏置。
- GeoBEV 是当前规划视觉入口；IRWM 保留独立的视觉/BEV 编码通路。IRWM 只更新最后 future decoder 层和 action fusion，共 **907,712** 个参数。

### 逐 token 的交互门控

每个交互层的 agent、ego、当前 BEV 路径分别使用：

```text
transformed = 原有 Post-LN 子步骤(h, 外部条件)
g = 2 * sigmoid(MLP(features))
h_new = (1 - g) * h + g * transformed
```

门控 MLP 为 `282 → 128 → 3`，输入为 block 输入的 LayerNorm 特征 256 维、几何 6 维与观测运动 20 维。每个 waypoint 输出三个门；输出层权重/偏置为零，初始化时 `g=1`，保留原交互。`g=0` 时精确绕过整个对应 Post-LN 子步骤。门控及插值用 FP32。

ego 单 key 注意力的 value/output 两个线性投影折叠为一个线性层，Q/K 移除。它在实数代数上等价，BF16 下两次与一次矩阵乘法可能有舍入差异。原未来分支的 scalar gate 和 point reliability gate 保持其原路径。

### 轨迹编解码

- 保留预训练归一化的仿射尺度：`u0=(delta0-3.55)/4.05`、`u1=delta1/3`。
- V11 的 GT 不再裁剪；预测、候选位置和 DDIM 更新使用扩展米制范围：第一维 `[-1,12]`、第二维 `[-5,5]`。
- 原始 CVCI2 的 12 层权重先严格加载并审计，再激活候选模块、删除 I12 和添加新门控。

本版本格式为 `cvci2res_geobev_plan_a_v11`，总可训练参数 **98,824,190**。

## 2. 完整训练配方

从原始 CVCI2、GeoBEV、IRWM 预训练权重开始，Res/V10/V11 新增模块随机初始化；不使用旧实验或校准模型的训练状态。第 2–10 轮仅恢复本次实验自己的 `last.pth`。

| 项目 | 配置 |
| --- | --- |
| GPU / batch | 4×H20-3e；12/卡；全局 48；梯度累积 1 |
| 预算 | 10 epoch × 587 更新 = 5870 更新 |
| 优化器 | fused AdamW；全局梯度范数裁剪 1.0 |
| 精度 | 主网络 BF16；IRWM FP32 / TF32；世界模型激活重计算 |
| 调度 | 500 更新线性 warmup；cosine 到峰值的 0.001 倍 |
| 数据加载 | workers 4/卡；prefetch 2；seed 2026 |
| 扩散时间步 | 50% 强制 t99，其他均匀 0–99；理论 P(t99)=50.5% |
| 冷启动 | 参与规划监督，IRWM 仍受历史有效性控制 |
| 增强 | GeoBEV 颜色增强概率 0.5、强度 0.1；future dropout 0.1 |
| 推理 | 一次去噪，t99 |

峰值学习率：

| 参数类别 | 峰值 LR |
| --- | ---: |
| 图像骨干 | `5e-6` |
| 其他预训练模块、depth heads | `1e-5` |
| 原 DiT 与折叠后的 ego 投影 | `2e-5` |
| 新 future、offset、BEV bridge、current readout | `1e-4` |
| 新交互门控 MLP body | `1e-4` |
| 新交互门控 MLP output | `1e-3` |
| future scalar gate / current point logits | `1e-3` |
| future distance strength | `1e-2` |
| IRWM 局部参数 | `1e-6` |

原参数矩阵 weight decay `1e-4`，新增非门控矩阵 `0.01`；按配置排除 norm、bias、embedding、向量参数等。新交互门控和独立控制参数不衰减，depth heads 的衰减为 0。实际 **22 个优化器参数组**及参数清单见 `recorded_config.json`。

### 损失

```text
L = weighted_mean(masked x0 MSE of 11 heads)
    + lambda(update) * final_self_t99_metric_position_L2
    + 0.2 * agent classification
    + 0.2 * agent box regression
    + 0.1 * current occupancy
    + 0.1 * GeoBEV depth

x0 raw head weights = [1,1,1,1,1,1,1,1,2,2,6] / 18
lambda(update) = 0.00095 * min(update / 500, 1)
```

位置项只监督最终 self 输出、且仅用于 t99 样本：

```text
valid_k = all labels of delta_1 .. delta_k are valid
e_k = ||cumsum(predicted metric delta)_k - cumsum(GT delta)_k||_2
w = [11,11,5,5,2,2]
per_sample = sum(w * valid * e) / sum(w * valid)
L_position = mean(per_sample over samples with a valid prefix)
```

这是各向同性、非平方的米制 L2。完整六点时分母 36，与 MeanPrefix 1/2/3s 的权重一致；部分标签保持同一组权重，只按有效权重和归一化。无效后缀在累计前清理，避免无效值污染前缀。

### 梯度比例监控与校准

初始系数来自 12 个真实训练批次的训练集校准，采用两位有效数字 `0.00095`。原始测量仅保留供追溯；正式训练从原始预训练权重重新初始化。

每 100 更新以及首步/轮末，复用同一噪声与条件，在独立参数副本上重放 planner，监控共享读出头与 I10/S11 上的位置项梯度相对加权 x0 梯度的比例。四卡先平均梯度向量，再计算范数；不重复 GeoBEV/IRWM、不推进数据流和 RNG。

`position_target_ratio=0.2` 是参考目标，参考区间为 10%–30%。**本次是固定系数加监控，没有自动系数控制器。** 预热后 64 次监测的中位比例为 30.98%，其中 34 次超过 30%；完整记录见 `results/position_gradient_probes.json`。

## 3. 配置、环境和权重

| 路径 | 内容 |
| --- | --- |
| [train.json](configs/v11_plan_a/train.json) | 正式启动配置，保持训练时原始字节 |
| [recorded_config.json](configs/v11_plan_a/recorded_config.json) | 第 10 轮实际配置、模型版本、分组、源码指纹和初始化审计 |
| [recorded_recipe.json](configs/v11_plan_a/recorded_recipe.json) | 协调器保存的相同完整配方 |
| [export_manifest.json](configs/v11_plan_a/export_manifest.json) | 导出与逐文件摘要 |
| [校准记录](configs/v11_plan_a/calibration/position_calibration.json) | 训练集初始梯度比例测量及系数来源 |
| [environment/](environment/README.md) | Python/依赖锁定和 H20 算子构建记录 |
| [启动前验证](docs/verification/verification_summary.json) | 数学检查、四卡短检查及精确恢复验证；记录时间早于正式训练 |
| [发布核验](docs/publication_checks.json) | 源码、参数、语法和 dry-run 核验 |

第 10 轮 `recorded_config.json` 的 `resume` 指向本次任务自己的 `last.pth`，用于逐轮连续训练；首轮从三份原始预训练权重开始。记录中的原服务器路径用于追溯。校准来源的绝对路径也原样保留，校准文件另存于上表的仓库内位置。

数据需包含 nuScenes trainval 图像、点云、元数据，以及本工程所用 `nuscenes_infos_train.pkl`、`nuscenes_infos_val.pkl`。原服务器数据入口为 `/mnt/workspace/dataSet/nuscene`。

三份原始预训练权重置于仓库同级 `Models/`，或用 `--models-dir` 指定：

```text
Models/
  best_endpoint_1step.pth
  geobev-r50-nuimage-cbgs.pth
  MMO_MSO_with_plan_tightly_coupled.pth
```

各权重 SHA256 在启动配置中，正式启动器会核对。仓库提供源码、环境记录、配置和小型结果记录；数据与权重另行准备。

## 4. 完整复现入口

Linux 下准备环境后，先检查命令（不导入 Torch、不占 GPU、不创建输出目录）：

```bash
python3 tools/run_plan_a.py \
  --data-root /mnt/workspace/dataSet/nuscene \
  --models-dir /mnt/workspace/nix3sgh/Models \
  --output experiments/v11_plan_a_reproduction \
  --dry-run
```

正式启动：

```bash
ENV/bin/python tools/run_plan_a.py \
  --data-root /mnt/workspace/dataSet/nuscene \
  --models-dir /mnt/workspace/nix3sgh/Models \
  --output experiments/v11_plan_a_reproduction
```

需四卡空闲，输出目录必须不存在。启动器按完整 5870 更新日程逐轮训练与验证，保存 `epoch_XX.pth`、`last.pth`、`epoch_metrics.json`、`best.json`。恢复的是本次任务的完整状态，不会每轮重置学习率。可通过 `--python`、`--world-root` 指定环境和 IRWM 配置目录。

单独全覆盖评估：

```bash
CVCI_WORLD_READOUT_STEPS=5 ENV/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 tools/eval_geobev_full_coverage.py \
  --adapter-checkpoint experiments/v11_plan_a_reproduction/run/epoch_10.pth \
  --output-dir experiments/v11_plan_a_eval \
  --base-checkpoint ../Models/best_endpoint_1step.pth \
  --world-checkpoint ../Models/MMO_MSO_with_plan_tightly_coupled.pth \
  --world-root world_configs \
  --data-root /mnt/workspace/dataSet/nuscene \
  --seed 2026 --ddim-steps 1 --layer-diagnostics \
  --workers 2 --prefetch-factor 2 --log-seconds 30
```

## 5. 训练结果

每轮遍历 6019 帧；未来六点全有效的 5119 帧计分，其中原滚动 4585 帧、冷启动 534 帧；未来不完整的 900 帧舍弃。按原滚动集合的 MeanPrefix L2 选择最佳 epoch。第 10 轮覆盖审计 `PASS`，不是 partial evaluation。

| 第 10 轮最终 self 输出 | Prefix 1s | Prefix 2s | Prefix 3s | MeanPrefix |
| --- | ---: | ---: | ---: | ---: |
| 全量 5119 帧 | 0.148895 | 0.302974 | 0.556691 | **0.336186** |
| 原滚动 4585 帧 | 0.147902 | 0.301820 | 0.554726 | **0.334816** |
| 冷启动 534 帧 | 0.157417 | 0.312877 | 0.573563 | **0.347952** |

单位均为米。Prefix 1/2/3s 为前 2/4/6 个累计位置误差均值，MeanPrefix 为三者均值。全量 Endpoint 1/2/3s 为 `0.198237 / 0.567372 / 1.258899 m`，应与 Prefix 分别比较。

相比上一版完成 10 轮的 [V10 Refined Cold](https://github.com/123mike-source/CVCI2_Res_V10_RefinedCold) 默认 H12，全量 MeanPrefix 从 `0.355696` 降至 `0.336186`，降低 **5.48%**。这是整套配置的单次训练对照，单项贡献需进一步实验区分。

完整记录：

- [10 轮指标](results/epoch_metrics.json)、[最佳轮次](results/best.json)、[完成状态](results/completion_state.json)。
- [第 10 轮完整指标](results/epoch_10/metrics.json)、[逐层诊断](results/epoch_10/layer_diagnostics.json)。逐层诊断共享最终层的滚动历史，不是各层独立滚动结果。
- [训练结果汇总](docs/training_results_summary.md)、[梯度比例汇总](results/position_gradient_summary.json)、[原始探针记录](results/position_gradient_probes.json)。

## 6. 来源与许可

基础工程：[CVCI2_Res](https://github.com/123mike-source/CVCI2_Res)、[CVCI2](https://github.com/123mike-source/CVCI2)。GeoBEV 派生模块保留 [LICENSE](geobev_encoder/LICENSE)、[NOTICE](geobev_encoder/NOTICE)。第三方源码保留各自原始声明。
