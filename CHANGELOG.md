# 更新日志

本文档记录项目的重要更新和变更。

## [2026-01-29 v2] - 简化追踪器，仅保留 ByteTrack

### 删除
- ❌ 删除 DeepSORT (ReID / NoReID) 追踪器
- ❌ 删除 SimpleIOU 追踪器
- ❌ 删除 Temporal 追踪器
- ❌ 删除多追踪器对比功能
- ❌ 移除 `--tracker` 命令行参数

### 修改
- 🔄 简化脚本，专注于 ByteTrack 单一追踪器
- 🔄 函数名由 `run_tracker` 改为 `run_bytetrack`
- 🔄 移除追踪器选择逻辑，直接使用 ByteTrack
- 🔄 更新文档，移除其他追踪器的说明
- 🔄 简化参数说明

### 优点
- ✅ 代码更简洁，易于维护
- ✅ 减少依赖（不再需要 deep-sort-realtime）
- ✅ 提高执行效率
- ✅ 用户使用更简单，无需选择追踪器

## [2026-01-29 v1] - 环境配置完善与计数逻辑优化

### 新增
- ✅ 添加 `.gitignore` 文件，排除虚拟环境、输出文件、模型文件等
- ✅ 添加自动安装脚本 `setup.bat` (Windows) 和 `setup.sh` (Linux/Mac)
- ✅ 添加详细的 `INSTALL.md` 安装指南
- ✅ 添加 `README_MODELS.md` 模型获取说明
- ✅ 添加 `LICENSE` 文件 (MIT)
- ✅ 添加 `.gitkeep` 文件保持目录结构（models, input_videos, output）
- ✅ 添加环境测试脚本 `test_environment.py`

### 修改
- 🔄 **计数逻辑优化**:
  - 移除视频中的区域标签显示（OUT, WAIT, ENTRY）
  - TOTAL 计数改为三条线穿越次数的平均值向下取整
  - 简化跟踪框标注，仅显示 ID
  - 优化左下角统计信息显示

- 📝 **文档更新**:
  - 优化 `README.md`，添加徽章、快速开始指南
  - 移除区域相关说明，改为线穿越计数说明
  - 添加详细的输出文件说明
  - 添加更多常见问题解答
  - 添加模型获取、贡献指南等章节

- 🔧 **依赖管理**:
  - 优化 `requirements.txt`，添加详细注释和版本说明
  - 明确标注 PyTorch 需单独安装
  - 添加可选的性能优化包说明

### 技术细节

#### 计数逻辑变更
**之前**: 显示 ENTRY、WAIT、OUT 三个区域的标签，TOTAL 计数基于有效轨迹判定

**现在**:
- 仅显示3条分隔线（Line 0, Line 1, Line 2）
- TOTAL = floor((Line0 + Line1 + Line2) / 3)
- 左下角显示各线穿越次数统计
- 跟踪框仅显示 ID，不显示区域路径

#### 文件修改位置
- `scripts/compare_trackers.py:552-585` - 视频标注逻辑

### 使用说明

#### 新用户快速开始
```bash
# 1. 克隆项目
git clone https://github.com/YOUR_USERNAME/pig_tracking_system.git
cd pig_tracking_system

# 2. 运行自动安装脚本
# Windows: setup.bat
# Linux/Mac: ./setup.sh

# 3. 准备模型和视频
# 将模型文件放到 models/ 目录
# 将视频文件放到 input_videos/ 目录

# 4. 运行处理
python scripts/batch_process_videos.py
```

#### 升级用户注意事项
如果你之前使用过旧版本，请注意：
1. 视频输出格式已变化，不再显示区域标签
2. 计数逻辑已改为线穿越平均值
3. 建议重新处理旧视频以获得新格式输出

---

## [Previous] - 初始版本

### 功能
- RF-DETR 目标检测
- ByteTrack / DeepSORT 跟踪
- 区域分析（ENTRY→WAIT→OUT）
- 轨迹分析和报告生成
