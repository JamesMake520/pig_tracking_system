@echo off
chcp 65001 >nul
echo ====================================
echo 猪只追踪系统 - 快速启动
echo ====================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1] 检查环境...
echo.

:: 检查虚拟环境
if not exist "venv\" (
    echo [2] 创建虚拟环境...
    python -m venv venv
    echo 虚拟环境创建完成
    echo.
)

:: 激活虚拟环境
echo [3] 激活虚拟环境...
call venv\Scripts\activate.bat
echo.

:: 检查依赖
echo [4] 检查依赖包...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到PyTorch，需要安装依赖包
    echo 请选择：
    echo   1. 自动安装依赖（需要联网，推荐）
    echo   2. 稍后手动安装
    choice /c 12 /n /m "请选择 [1/2]: "
    if errorlevel 2 goto :manual
    if errorlevel 1 goto :auto_install
)

echo 依赖包已安装
goto :check_model

:auto_install
echo.
echo [5] 安装PyTorch (CUDA 11.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.
echo [6] 安装其他依赖...
pip install -r requirements.txt
echo.
echo 依赖安装完成！
goto :check_model

:manual
echo.
echo 请手动运行以下命令安装依赖：
echo   1. 激活虚拟环境: venv\Scripts\activate
echo   2. 安装PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo   3. 安装依赖: pip install -r requirements.txt
echo.
pause
exit /b 0

:check_model
echo.
echo [检查] 验证模型文件...
if not exist "models\checkpoint_best_ema.pth" (
    echo [警告] 未找到模型文件: models\checkpoint_best_ema.pth
    echo 请将训练好的模型文件放到 models\ 目录下
    echo.
)

dir /b input_videos\*.mp4 >nul 2>&1
if errorlevel 1 (
    echo [提示] input_videos\ 目录为空
    echo 请将要处理的视频文件放到 input_videos\ 目录下
    echo.
)

echo ====================================
echo 环境准备完成！
echo ====================================
echo.
echo 使用方法：
echo   1. 将视频文件放到 input_videos\ 目录
echo   2. 将模型文件放到 models\ 目录
echo   3. 运行批量处理: python scripts\batch_process_videos.py
echo.
echo 或者查看 README.md 了解更多信息
echo.
pause
