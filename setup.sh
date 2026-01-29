#!/bin/bash
# 猪只追踪系统 - 自动安装脚本 (Linux/Mac)

set -e

echo "================================================"
echo "猪只追踪系统 - 环境配置脚本"
echo "================================================"
echo ""

# 检查 Python 版本
echo "[1/6] 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  当前 Python 版本: $python_version"

# 检查是否满足最低版本要求 (3.8)
required_version="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "  [ERROR] Python 版本过低，需要 3.8 或更高版本"
    exit 1
fi
echo "  [OK] Python 版本满足要求"
echo ""

# 创建虚拟环境
echo "[2/6] 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "  虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "  [OK] 虚拟环境创建成功"
fi
echo ""

# 激活虚拟环境
echo "[3/6] 激活虚拟环境..."
source venv/bin/activate
echo "  [OK] 虚拟环境已激活"
echo ""

# 升级 pip
echo "[4/6] 升级 pip..."
pip install --upgrade pip
echo ""

# 安装 PyTorch (根据系统自动选择)
echo "[5/6] 安装 PyTorch..."
echo "  请选择安装方式:"
echo "  1) CUDA 11.8 (GPU)"
echo "  2) CUDA 12.1 (GPU)"
echo "  3) CPU only"
read -p "  请输入选项 [1-3]: " cuda_choice

case $cuda_choice in
    1)
        echo "  安装 PyTorch (CUDA 11.8)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "  安装 PyTorch (CUDA 12.1)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "  安装 PyTorch (CPU)..."
        pip install torch torchvision torchaudio
        ;;
    *)
        echo "  [ERROR] 无效选项"
        exit 1
        ;;
esac
echo ""

# 安装依赖包
echo "[6/6] 安装项目依赖..."
pip install -r requirements.txt
echo "  [OK] 依赖安装完成"
echo ""

# 创建必要目录
echo "创建必要目录..."
mkdir -p input_videos
mkdir -p output
mkdir -p models
echo "  [OK] 目录创建完成"
echo ""

# 运行环境测试
echo "================================================"
echo "运行环境测试..."
echo "================================================"
python test_environment.py
echo ""

echo "================================================"
echo "安装完成！"
echo "================================================"
echo ""
echo "下一步:"
echo "  1. 将模型文件 (.pth) 放到 models/ 目录"
echo "  2. 将视频文件放到 input_videos/ 目录"
echo "  3. 运行: python scripts/batch_process_videos.py"
echo ""
echo "重新激活虚拟环境: source venv/bin/activate"
echo "================================================"
