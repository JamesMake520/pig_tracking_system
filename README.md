# 猪只追踪与计数系统

基于 RF-DETR 检测模型和 ByteTrack 跟踪算法的猪只自动追踪和计数系统。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ 功能特点

- 🎯 **高精度检测**: 使用 RF-DETR 进行高精度目标检测
- 🔄 **多种跟踪算法**: 支持 ByteTrack、DeepSORT 等多种跟踪算法
- 📈 **线穿越计数**: 使用3条分隔线统计穿越次数，最终计数为平均值向下取整
- 📹 **可视化输出**: 生成带标注的追踪视频，显示ID和线穿越次数
- 📝 **详细报告**: 生成ID事件日志、状态变化记录、轨迹分析报告
- 🚫 **智能过滤**: 自动过滤蓝色物体

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/pig_tracking_system.git
cd pig_tracking_system
```

### 2. 一键安装

**Windows 用户**：
```bash
setup.bat
```

**Linux/Mac 用户**：
```bash
chmod +x setup.sh
./setup.sh
```

安装脚本会自动完成以下步骤：
- ✅ 创建 Python 虚拟环境
- ✅ 安装 PyTorch（支持 CUDA 11.8/12.1/CPU）
- ✅ 安装所有依赖包
- ✅ 创建必要的目录结构
- ✅ 运行环境测试

### 3. 准备模型和视频

```bash
# 将模型文件放到 models 目录
models/checkpoint_best_ema.pth

# 将视频文件放到 input_videos 目录
input_videos/your_video.mp4
```

> 📝 如果没有模型文件，请联系项目维护者或查看 [模型获取说明](#模型获取)

### 4. 运行处理

```bash
# 激活虚拟环境
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 批量处理所有视频
python scripts/batch_process_videos.py
```

完成！处理结果会保存在 `output/` 目录。

> 💡 详细安装步骤和问题解决请查看 [INSTALL.md](INSTALL.md)

## 目录结构

```
pig_tracking_system/
├── scripts/
│   ├── batch_process_videos.py      # 批量处理脚本
│   └── compare_trackers.py          # 核心追踪脚本
├── rfdetr/                           # RF-DETR 检测模块
├── third_party/
│   └── OC_SORT/trackers/             # ByteTrack等跟踪器
├── models/                           # 模型文件目录（需自行放置）
├── input_videos/                     # 输入视频目录
├── output/                           # 输出结果目录
├── requirements.txt                  # 依赖包列表
└── README.md                         # 本文件
```

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐使用 GPU)
- Windows 10/11 或 Linux

## 📦 安装步骤

### 方法 1：自动安装（推荐）

使用提供的安装脚本一键完成环境配置：

```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh && ./setup.sh
```

### 方法 2：手动安装

如果自动安装失败，请参考以下步骤手动安装：

#### 1. 创建虚拟环境

```bash
# 使用 venv (推荐)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 或使用 conda
conda create -n pig_tracking python=3.10
conda activate pig_tracking
```

#### 2. 安装 PyTorch

根据你的 CUDA 版本选择：

```bash
# CUDA 11.8 (RTX 30/40 系列推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (新显卡)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (无 GPU)
pip install torch torchvision torchaudio
```

#### 3. 安装依赖包

```bash
pip install -r requirements.txt
```

#### 4. 准备模型文件

将训练好的 RF-DETR 模型文件放到 `models/` 目录：
```
models/checkpoint_best_ema.pth
```

> 💡 详细安装说明和问题解决请查看 [INSTALL.md](INSTALL.md)

## 使用方法

### 方法1：批量处理（推荐）

批量处理 `input_videos/` 目录下的所有视频：

```bash
python scripts/batch_process_videos.py
```

### 方法2：单个视频处理

处理单个视频文件：

```bash
python scripts/compare_trackers.py \
    --video_path "input_videos/test.mp4" \
    --model_path "models/checkpoint_best_ema.pth" \
    --output_dir "output/test_result" \
    --tracker "ByteTrack" \
    --no_timestamp
```

**参数说明**：
- `--video_path`: 输入视频路径
- `--model_path`: RF-DETR 模型路径
- `--output_dir`: 输出目录
- `--tracker`: 跟踪算法 (ByteTrack/DeepSORT_ReID/DeepSORT_NoReID/SimpleIOU)
- `--no_timestamp`: 不在输出目录名中添加时间戳
- `--out_ratio`: 第一区域占比 (默认0.45，用于确定Line 0和Line 1的位置)
- `--wait_ratio`: 第二区域占比 (默认0.25，用于确定Line 2的位置)

### 计数逻辑说明

系统使用3条分隔线统计猪只穿越次数：
- **Line 0**: 位于画面左侧 (out_ratio/2 处)
- **Line 1**: 位于画面中部偏左 (out_ratio 处)
- **Line 2**: 位于画面中部偏右 (out_ratio + wait_ratio 处)

**最终计数** = floor((Line0穿越次数 + Line1穿越次数 + Line2穿越次数) / 3)

即三条线穿越次数的平均值向下取整。

## 输出文件

每个视频处理后会在输出目录生成以下文件：

1. **ByteTrack_result.mp4** - 带标注的追踪视频
   - 显示检测框和ID
   - 显示3条分隔线
   - 显示每条线的穿越次数
   - 显示最终计数（TOTAL，三线平均值向下取整）

2. **ByteTrack_id_events.csv** - ID事件日志
   - 记录每个ID的出现、消失、状态变化

3. **ByteTrack_state_changes.txt** - 详细状态变化记录
   - 每个ID的完整轨迹分析

4. **ByteTrack_trajectory_report.csv** - 轨迹分析报告
   - 统计每个ID的运动数据

5. **comparison_summary.txt** - 总结报告
   - 整体统计信息

## 常见问题

### Q1: 如何调整检测阈值？
编辑 `compare_trackers.py`，修改 `conf_thresh` 参数（默认0.5）

### Q2: GPU内存不足怎么办？
1. 减小视频分辨率
2. 降低检测阈值
3. 使用CPU模式（较慢）

### Q3: 如何批量处理大量视频？
使用 `batch_process_videos.py`，它会自动逐个处理所有视频

### Q4: 如何调整分隔线的位置？
通过 `--out_ratio` 和 `--wait_ratio` 参数调整：
```bash
python scripts/compare_trackers.py \
    --video_path "input_videos/test.mp4" \
    --out_ratio 0.5 \
    --wait_ratio 0.3
```

### Q5: 最终计数是如何计算的？
最终计数 = floor((Line0 + Line1 + Line2) / 3)，即三条线穿越次数的平均值向下取整

## 🔧 技术栈

- **检测模型**：RF-DETR
- **跟踪算法**：ByteTrack / DeepSORT
- **深度学习框架**：PyTorch 2.0+
- **计算机视觉**：OpenCV
- **数据处理**：NumPy, Pandas

## 📥 模型获取

本项目需要预训练的 RF-DETR 模型文件。由于模型文件较大（>100MB），不包含在 Git 仓库中。

### 获取方式

1. **联系项目维护者**: 如果你是团队成员，请联系管理员获取模型文件
2. **使用自己的模型**: 如果你有自己训练的 RF-DETR 模型，直接放到 `models/` 目录即可

### 放置位置

```
pig_tracking_system/
└── models/
    └── checkpoint_best_ema.pth  # 你的模型文件（名称可自定义）
```

如果使用不同名称的模型，运行时请指定：

```bash
python scripts/compare_trackers.py --model_path "models/your_model.pth"
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 项目结构说明

```
pig_tracking_system/
├── scripts/                    # 主要脚本
│   ├── batch_process_videos.py    # 批量处理
│   └── compare_trackers.py        # 核心追踪脚本
├── rfdetr/                     # RF-DETR 检测模块
├── third_party/                # 第三方库
│   └── OC_SORT/trackers/          # 跟踪算法
├── models/                     # 模型文件 (需自行放置)
├── input_videos/               # 输入视频目录
├── output/                     # 输出结果目录
├── docs/                       # 文档
├── test_environment.py         # 环境测试脚本
├── setup.bat / setup.sh        # 自动安装脚本
├── requirements.txt            # 依赖列表
├── INSTALL.md                  # 详细安装指南
└── README.md                   # 本文件
```

## 🐛 问题反馈

如遇到问题，请：

1. 首先查看 [INSTALL.md](INSTALL.md) 中的常见问题
2. 运行 `python test_environment.py` 检查环境配置
3. 在 Issues 中搜索是否有类似问题
4. 提交新 Issue 时请附上：
   - 错误信息完整截图
   - 系统信息（OS、Python 版本、CUDA 版本）
   - `test_environment.py` 的输出

## 📚 相关资源

- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [OpenCV 文档](https://docs.opencv.org/)
- [ByteTrack 论文](https://arxiv.org/abs/2110.06864)
- [DETR 系列论文](https://arxiv.org/abs/2010.04159)

## 📝 许可证

本项目仅供学习和研究使用。

## 👏 致谢

感谢以下开源项目：
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [RF-DETR](https://github.com/Peterande/RT-DETR)
- [Supervision](https://github.com/roboflow/supervision)

---

**最后更新**: 2026-01-29

如有问题或建议，欢迎提 Issue 或 Pull Request！
