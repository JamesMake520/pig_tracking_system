#!/bin/bash
# 猪只追踪系统 - 自动安装脚本 (Linux/Mac)

set -e

: "${PIG_CUDA:=auto}"

resolve_cuda_choice() {
    local choice="${PIG_CUDA,,}"
    case "$choice" in
        cu118|11.8)
            CUDA_CHOICE="cu118"
            CUDA_VERSION="11.8"
            CUDA_VERSION_SOURCE="env:PIG_CUDA"
            return
            ;;
        cu121|12.1)
            CUDA_CHOICE="cu121"
            CUDA_VERSION="12.1"
            CUDA_VERSION_SOURCE="env:PIG_CUDA"
            return
            ;;
        cpu)
            CUDA_CHOICE="cpu"
            CUDA_VERSION="CPU"
            CUDA_VERSION_SOURCE="env:PIG_CUDA"
            return
            ;;
        auto|"")
            ;;
        *)
            ;;
    esac

    detect_cuda_version
    if [[ -n "$CUDA_VERSION" ]]; then
        map_cuda_version "$CUDA_VERSION"
    else
        CUDA_CHOICE="cpu"
    fi
}

detect_cuda_version() {
    CUDA_VERSION=""
    CUDA_VERSION_SOURCE=""

    if [[ -n "${CUDA_PATH:-}" && -f "${CUDA_PATH}/version.txt" ]]; then
        CUDA_VERSION="$(grep -i "CUDA Version" "${CUDA_PATH}/version.txt" | awk '{print $3}' | head -n1)"
        if [[ -n "$CUDA_VERSION" ]]; then
            CUDA_VERSION_SOURCE="CUDA_PATH"
            return
        fi
    fi

    if [[ -n "${CUDA_HOME:-}" && -f "${CUDA_HOME}/version.txt" ]]; then
        CUDA_VERSION="$(grep -i "CUDA Version" "${CUDA_HOME}/version.txt" | awk '{print $3}' | head -n1)"
        if [[ -n "$CUDA_VERSION" ]]; then
            CUDA_VERSION_SOURCE="CUDA_HOME"
            return
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
        if [[ -n "$CUDA_VERSION" ]]; then
            CUDA_VERSION_SOURCE="nvidia-smi"
            return
        fi
    fi

    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -n1)"
        if [[ -n "$CUDA_VERSION" ]]; then
            CUDA_VERSION_SOURCE="nvcc"
            return
        fi
    fi
}

map_cuda_version() {
    local ver="$1"
    local major="${ver%%.*}"
    local rest="${ver#*.}"
    local minor="${rest%%.*}"

    if [[ "$ver" == "$major" ]]; then
        minor=0
    fi

    if (( major > 12 )); then
        CUDA_CHOICE="cu121"
        return
    fi
    if (( major == 12 )); then
        if (( minor >= 1 )); then
            CUDA_CHOICE="cu121"
        else
            CUDA_CHOICE="cu118"
        fi
        return
    fi
    if (( major == 11 )); then
        if (( minor >= 8 )); then
            CUDA_CHOICE="cu118"
        else
            CUDA_CHOICE="cpu"
        fi
        return
    fi

    CUDA_CHOICE="cpu"
}

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
python3 -m pip install --upgrade pip
echo ""

# 安装 PyTorch（自动检测 CUDA）
echo "[5/6] 安装 PyTorch..."
CUDA_CHOICE=""
CUDA_VERSION=""
CUDA_VERSION_SOURCE=""
resolve_cuda_choice

case "$CUDA_CHOICE" in
    cu118)
        echo "  检测到 CUDA 版本: $CUDA_VERSION ($CUDA_VERSION_SOURCE)"
        echo "  安装 PyTorch (CUDA 11.8)..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    cu121)
        echo "  检测到 CUDA 版本: $CUDA_VERSION ($CUDA_VERSION_SOURCE)"
        echo "  安装 PyTorch (CUDA 12.1)..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        if [[ "${PIG_CUDA,,}" == "cpu" ]]; then
            echo "  已指定 PIG_CUDA=cpu，安装 CPU 版本"
        elif [[ -n "$CUDA_VERSION" ]]; then
            echo "  [WARN] CUDA 版本不满足 11.8+（检测到: $CUDA_VERSION）。将安装 CPU 版本"
        else
            echo "  [WARN] 未检测到 NVIDIA CUDA，将安装 CPU 版本"
        fi
        python3 -m pip install torch torchvision torchaudio
        ;;
esac
echo ""

# 安装依赖包
echo "[6/6] 安装项目依赖..."
python3 -m pip install -r requirements.txt
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
