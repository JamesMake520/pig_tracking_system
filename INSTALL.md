# 安装指南

本文档提供详细的安装步骤和常见问题解决方案。

## 目录

- [系统要求](#系统要求)
- [快速安装](#快速安装)
- [手动安装](#手动安装)
- [常见问题](#常见问题)
- [验证安装](#验证安装)

## 系统要求

### 硬件要求

- **内存**: 8GB RAM（推荐 16GB+）
- **显卡**:
  - NVIDIA GPU（推荐）：显存 4GB+（推荐 6GB+）
  - 支持 CUDA 11.8 或 12.1
  - 也可以使用 CPU，但速度会慢很多

### 软件要求

- **操作系统**: Windows 10/11、Linux（Ubuntu 20.04+）、macOS
- **Python**: 3.8 或更高版本（推荐 3.10）
- **CUDA**: 11.8 或 12.1（GPU 版本需要）
- **Git**: 用于克隆仓库

### 检查 CUDA 版本

```bash
# 检查 NVIDIA 驱动和 CUDA 版本
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

如果没有安装 CUDA，请访问 [NVIDIA CUDA 官网](https://developer.nvidia.com/cuda-downloads) 下载安装。

## 快速安装

### Windows 用户

1. 克隆仓库
```bash
git clone https://github.com/YOUR_USERNAME/pig_tracking_system.git
cd pig_tracking_system
```

2. 运行自动安装脚本
```bash
setup.bat
```

3. 按提示选择 CUDA 版本（1=CUDA 11.8, 2=CUDA 12.1, 3=CPU）

### Linux/Mac 用户

1. 克隆仓库
```bash
git clone https://github.com/YOUR_USERNAME/pig_tracking_system.git
cd pig_tracking_system
```

2. 运行自动安装脚本
```bash
chmod +x setup.sh
./setup.sh
```

3. 按提示选择 CUDA 版本

## 手动安装

如果自动安装脚本失败，可以按照以下步骤手动安装。

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/pig_tracking_system.git
cd pig_tracking_system
```

### 2. 创建虚拟环境

#### 使用 venv（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 使用 Conda

```bash
conda create -n pig_tracking python=3.10
conda activate pig_tracking
```

### 3. 升级 pip

```bash
python -m pip install --upgrade pip
```

### 4. 安装 PyTorch

选择适合你系统的 PyTorch 版本：

#### GPU 版本 - CUDA 11.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 版本 - CUDA 12.1

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU 版本

```bash
pip install torch torchvision torchaudio
```

### 5. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 6. 准备模型文件

将 RF-DETR 模型文件放到 `models/` 目录：

```
pig_tracking_system/
└── models/
    └── checkpoint_best_ema.pth  # 你的模型文件
```

如果没有模型文件，请联系项目维护者获取。

### 7. 准备输入视频

将要处理的视频文件放到 `input_videos/` 目录：

```
pig_tracking_system/
└── input_videos/
    ├── video1.mp4
    └── video2.mp4
```

## 常见问题

### Q1: 安装 PyTorch 时速度很慢

**解决方案**：使用国内镜像加速

```bash
# 使用清华镜像
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: Windows 上安装 Cython 相关包失败

**错误信息**：`error: Microsoft Visual C++ 14.0 or greater is required`

**解决方案**：安装 Microsoft C++ Build Tools

1. 下载 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 安装时选择 "C++ build tools"
3. 重新运行 `pip install -r requirements.txt`

### Q3: 提示 `ImportError: DLL load failed`

**解决方案**：

1. 确保安装了正确的 CUDA 版本
2. 检查 NVIDIA 驱动是否最新
3. 尝试重新安装 PyTorch

### Q4: GPU 不可用，提示 `CUDA not available`

**解决方案**：

```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
```

如果返回 False：
1. 检查是否安装了 GPU 版本的 PyTorch
2. 检查 CUDA 版本是否匹配
3. 更新 NVIDIA 驱动

### Q5: 内存不足错误

**解决方案**：

1. 减少 batch size（如果脚本支持）
2. 降低视频分辨率
3. 使用更小的模型
4. 关闭其他占用 GPU 的程序

### Q6: OpenCV 导入错误

**错误信息**：`ImportError: libGL.so.1: cannot open shared object file`

**解决方案**（Linux）：

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
```

### Q7: 找不到模型文件

**错误信息**：`FileNotFoundError: models/checkpoint_best_ema.pth`

**解决方案**：

1. 确保模型文件在 `models/` 目录
2. 检查文件名是否正确
3. 或在运行时指定完整路径：
```bash
python scripts/compare_trackers.py --model_path "path/to/your/model.pth"
```

## 验证安装

运行环境测试脚本：

```bash
python test_environment.py
```

**期望输出**：

```
============================================================
环境检查 - 猪只追踪系统
============================================================

[1] Python版本
[OK] Python 3.10.x

[2] 核心依赖包
[OK] PyTorch
[OK] OpenCV
[OK] NumPy
[OK] Pandas
[OK] tqdm
[OK] supervision
[OK] deep-sort-realtime

[3] GPU状态
[OK] GPU可用: NVIDIA GeForce RTX 3080
  CUDA版本: 11.8

[4] 文件检查
[OK] 找到模型文件: checkpoint_best_ema.pth

============================================================
[SUCCESS] 环境配置完成！可以开始使用

建议运行:
  python scripts/batch_process_videos.py
============================================================
```

如果所有检查都显示 `[OK]`，说明环境配置成功！

## 快速开始

安装完成后，可以开始处理视频：

```bash
# 批量处理 input_videos 目录下的所有视频
python scripts/batch_process_videos.py

# 或处理单个视频
python scripts/compare_trackers.py \
    --video_path "input_videos/test.mp4" \
    --model_path "models/checkpoint_best_ema.pth" \
    --output_dir "output/test_result"
```

更多使用说明请查看 [README.md](README.md)。

## 获取帮助

如果遇到其他问题：

1. 查看 [README.md](README.md) 中的常见问题
2. 检查项目 Issues 页面
3. 提交新的 Issue 并附上错误信息

---

**最后更新**: 2026-01-29
