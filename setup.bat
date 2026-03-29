@echo off
REM 猪只追踪系统 - 自动安装脚本 (Windows)

chcp 65001 > nul
setlocal enabledelayedexpansion

echo ================================================
echo 猪只追踪系统 - 环境配置脚本
echo ================================================
echo.

REM 检查 Python 版本
echo [1/6] 检查 Python 版本...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] 未找到 Python，请先安装 Python 3.8+
    echo   下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo   当前 Python 版本: %PYTHON_VERSION%

REM 检查版本是否满足要求
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python 版本过低，需要 3.8 或更高版本
    pause
    exit /b 1
)
echo   [OK] Python 版本满足要求
echo.

REM 创建虚拟环境
echo [2/6] 创建虚拟环境...
if exist venv (
    echo   虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    echo   [OK] 虚拟环境创建成功
)
echo.

REM 激活虚拟环境
echo [3/6] 激活虚拟环境...
call venv\Scripts\activate.bat
echo   [OK] 虚拟环境已激活
echo.

REM 升级 pip
echo [4/6] 升级 pip...
python -m pip install --upgrade pip
echo.

REM 安装 PyTorch
echo [5/6] 安装 PyTorch...
echo   请选择安装方式:
echo   1) CUDA 11.8 (GPU - 推荐 RTX 30/40 系列)
echo   2) CUDA 12.1 (GPU - 新显卡)
echo   3) CPU only (无 GPU 或测试用)
echo.
set /p cuda_choice="  请输入选项 [1-3]: "

if "%cuda_choice%"=="1" (
    echo   安装 PyTorch ^(CUDA 11.8^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="2" (
    echo   安装 PyTorch ^(CUDA 12.1^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%cuda_choice%"=="3" (
    echo   安装 PyTorch ^(CPU^)...
    pip install torch torchvision torchaudio
) else (
    echo   [ERROR] 无效选项
    pause
    exit /b 1
)
echo.

REM 安装依赖包
echo [6/6] 安装项目依赖...
pip install -r requirements.txt
echo   [OK] 依赖安装完成
echo.

REM 创建必要目录
echo 创建必要目录...
if not exist input_videos mkdir input_videos
if not exist output mkdir output
if not exist models mkdir models
echo   [OK] 目录创建完成
echo.

REM 运行环境测试
echo ================================================
echo 运行环境测试...
echo ================================================
python test_environment.py
echo.

echo ================================================
echo 安装完成！
echo ================================================
echo.
echo 下一步:
echo   1. 将模型文件 (.pth) 放到 models\ 目录
echo   2. 将视频文件放到 input_videos\ 目录
echo   3. 运行: python scripts\batch_process_videos.py
echo.
echo 重新激活虚拟环境: venv\Scripts\activate.bat
echo ================================================
echo.
pause
