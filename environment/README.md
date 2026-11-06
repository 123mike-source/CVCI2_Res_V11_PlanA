# Recorded H20 environment

本目录记录 V11 方案 A 完整 10 轮训练所用的环境。

- Linux x86_64、Python 3.8.20。
- Torch 2.0.0+cu118、torchvision 0.15.1+cu118。
- MMDetection 2.28.2、MMCV-full 1.7.2，按 H20 `sm_90` 构建。
- NumPy 1.23.5、SciPy 1.10.1、nuScenes-devkit 1.1.10。
- `recorded_requirements.lock.txt` 为服务器实际 pip freeze 记录。
- 两份 JSON 保留 MMCV 和 CUDA 工具链的编译记录及原服务器路径。

## 环境重建

在独立 Linux 工作目录运行：

```bash
python3 setup_env.py
ENV/bin/python tools/check_environment.py
```

`setup_env.py` 安装项目内 Python 3.8 及依赖、配置 OpenCV 动态库，并调用 `tools/build_mmcv_sm90.py` 构建 MMCV CUDA 算子。工具链下载 CUDA 11.8 组件及 GCC 11，需要 `apt-get`、`dpkg-deb`、`curl`、NVIDIA 驱动和网络。

本工程使用 MMDetection/MMCV 1.x API，以及 `vendor/irwm_runtime/` 中的 IRWM 兼容运行时。梯度监控使用 PyTorch 2.0 的 `torch.func.functional_call`。原实验使用 4 张 H20-3e，每卡 batch 12，IRWM 局部更新。

配置入口为 [train.json](../configs/v11_plan_a/train.json)，完整启动方法见 [仓库首页](../README.md)。`tools/run_plan_a.py --dry-run` 只生成命令，不加载 Torch 或占用 GPU。
