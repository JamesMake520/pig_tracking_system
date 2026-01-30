@echo off
REM 猪只追踪系统 - 自动安装脚本 (Windows)

chcp 65001 > nul
setlocal enabledelayedexpansion

REM 环境变量配置（可选）
REM   - PIG_CUDA: auto|cu121|cu118|cpu|12.1|11.8
REM   - PIP_INDEX_URL / PIP_EXTRA_INDEX_URL / PIP_TRUSTED_HOST: pip 镜像（可选）
if not defined PIG_CUDA set "PIG_CUDA=auto"

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
set "CUDA_CHOICE="
set "CUDA_VERSION="
set "CUDA_VERSION_SOURCE="
call :resolve_cuda_choice

if /i "%CUDA_CHOICE%"=="cu118" (
    echo   检测到 CUDA 版本: !CUDA_VERSION! (!CUDA_VERSION_SOURCE!)
    echo   安装 PyTorch ^(CUDA 11.8^)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if /i "%CUDA_CHOICE%"=="cu121" (
    echo   检测到 CUDA 版本: !CUDA_VERSION! (!CUDA_VERSION_SOURCE!)
    echo   安装 PyTorch ^(CUDA 12.1^)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    if /i "%PIG_CUDA%"=="cpu" (
        echo   已指定 PIG_CUDA=cpu，安装 CPU 版本
    ) else if defined CUDA_VERSION (
        echo   [WARN] CUDA 版本不满足 11.8+（检测到: !CUDA_VERSION!）。将安装 CPU 版本
    ) else (
        echo   [WARN] 未检测到 NVIDIA CUDA，将安装 CPU 版本
    )
    python -m pip install torch torchvision torchaudio
)
echo.

REM 安装依赖包
echo [6/6] 安装项目依赖...
python -m pip install -r requirements.txt
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
exit /b 0

:resolve_cuda_choice
REM 允许通过环境变量覆盖（优先级最高）
if /i "%PIG_CUDA%"=="cu118" set "CUDA_CHOICE=cu118" & set "CUDA_VERSION=11.8" & set "CUDA_VERSION_SOURCE=env:PIG_CUDA" & goto :eof
if /i "%PIG_CUDA%"=="cu121" set "CUDA_CHOICE=cu121" & set "CUDA_VERSION=12.1" & set "CUDA_VERSION_SOURCE=env:PIG_CUDA" & goto :eof
if /i "%PIG_CUDA%"=="11.8" set "CUDA_CHOICE=cu118" & set "CUDA_VERSION=11.8" & set "CUDA_VERSION_SOURCE=env:PIG_CUDA" & goto :eof
if /i "%PIG_CUDA%"=="12.1" set "CUDA_CHOICE=cu121" & set "CUDA_VERSION=12.1" & set "CUDA_VERSION_SOURCE=env:PIG_CUDA" & goto :eof
if /i "%PIG_CUDA%"=="cpu" set "CUDA_CHOICE=cpu" & set "CUDA_VERSION=CPU" & set "CUDA_VERSION_SOURCE=env:PIG_CUDA" & goto :eof

call :detect_cuda_version
if defined CUDA_VERSION (
    call :map_cuda_version "!CUDA_VERSION!"
    goto :eof
)

set "CUDA_CHOICE=cpu"
goto :eof

:detect_cuda_version
set "CUDA_VERSION="
set "CUDA_VERSION_SOURCE="

REM 优先从 CUDA_PATH 获取版本
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\version.txt" (
        for /f "tokens=3" %%A in ('findstr /i "CUDA Version" "%CUDA_PATH%\version.txt"') do set "CUDA_VERSION=%%A"
        if defined CUDA_VERSION set "CUDA_VERSION_SOURCE=CUDA_PATH"
    )
)

REM 其次从 nvidia-smi 获取版本（若已安装驱动）
if not defined CUDA_VERSION (
    for /f "tokens=3 delims=:" %%A in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do set "CUDA_VERSION=%%A"
    for /f "tokens=* delims= " %%A in ("!CUDA_VERSION!") do set "CUDA_VERSION=%%A"
    if defined CUDA_VERSION set "CUDA_VERSION_SOURCE=nvidia-smi"
)
goto :eof

:map_cuda_version
set "VER=%~1"
for /f "tokens=1,2 delims=." %%A in ("%VER%") do (
    set /a CUDA_MAJOR=%%A
    set /a CUDA_MINOR=0
    if not "%%B"=="" set /a CUDA_MINOR=%%B
)

if not defined CUDA_MAJOR set "CUDA_CHOICE=cpu" & goto :eof

if !CUDA_MAJOR! GTR 12 (
    set "CUDA_CHOICE=cu121"
    goto :eof
)

if !CUDA_MAJOR! EQU 12 (
    if !CUDA_MINOR! GEQ 1 (
        set "CUDA_CHOICE=cu121"
    ) else (
        set "CUDA_CHOICE=cu118"
    )
    goto :eof
)

if !CUDA_MAJOR! EQU 11 (
    if !CUDA_MINOR! GEQ 8 (
        set "CUDA_CHOICE=cu118"
    ) else (
        set "CUDA_CHOICE=cpu"
    )
    goto :eof
)

set "CUDA_CHOICE=cpu"
goto :eof
