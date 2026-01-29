#!/usr/bin/env python3
"""
快速测试脚本 - 用于验证环境是否正确配置
"""
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    if version < (3, 8):
        print("  [WARNING] Python版本过低，建议使用3.8+")
        return False
    return True

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} - 未安装")
        return False

def check_gpu():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[OK] GPU可用: {gpu_name}")
            print(f"  CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("[FAIL] GPU不可用 (将使用CPU，速度较慢)")
            return False
    except:
        return False

def check_files():
    """检查必要的文件"""
    project_root = Path(__file__).parent

    # 检查模型文件
    model_dir = project_root / "models"
    model_files = list(model_dir.glob("*.pth"))
    if model_files:
        print(f"[OK] 找到模型文件: {model_files[0].name}")
        return True
    else:
        print(f"[FAIL] 未找到模型文件 (.pth) 在 {model_dir}")
        return False

def main():
    print("="*60)
    print("环境检查 - 猪只追踪系统")
    print("="*60)
    print()

    print("[1] Python版本")
    check_python_version()
    print()

    print("[2] 核心依赖包")
    packages = [
        ("PyTorch", "torch"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("supervision", "supervision"),
        ("Cython", "Cython"),
    ]

    all_installed = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_installed = False
    print()

    print("[3] GPU状态")
    has_gpu = check_gpu()
    print()

    print("[4] 文件检查")
    has_model = check_files()
    print()

    print("="*60)
    if all_installed and has_model:
        print("[SUCCESS] 环境配置完成！可以开始使用")
        print("\n建议运行:")
        print("  python scripts/batch_process_videos.py")
    else:
        if not all_installed:
            print("[WARNING] 部分依赖包未安装")
        if not has_model:
            print("[WARNING] 请将模型文件放到 models/ 目录")
    print("="*60)

if __name__ == "__main__":
    main()
