@echo off

REM Install mqbench because it cannot be installed from pypi
pip install git+https://github.com/audreyeternal/MQBench.git

REM Install development head of efficientbioai
cd EfficientBioAI
pip install -e .[%1]
cd ..


REM Install forked cellpose to fix some graph tracing problems.
pip install git+https://github.com/audreyeternal/cellpose.git


REM Check CUDA version and install PyTorch 1.10.0 accordingly
SET CUDA_VER=
FOR /F "tokens=5" %%i IN ('nvcc --version 2^>nul ^| find "release"') DO (
    SET CUDA_VER=%%i
)

IF NOT DEFINED CUDA_VER (
    ECHO No CUDA found, installing CPU version of PyTorch...
    pip install torch==1.10.0
    GOTO END
)

REM Extract major version of CUDA
FOR /F "tokens=1 delims=." %%i IN ("%CUDA_VER%") DO (
    SET CUDA_MAJOR_VER=%%i
)

IF "%CUDA_MAJOR_VER%"=="10" (
    SET TORCH_URL=https://download.pytorch.org/whl/cu102/
) ELSE IF "%CUDA_MAJOR_VER%"=="11" (
    SET TORCH_URL=https://download.pytorch.org/whl/cu113/
) ELSE (
    SET TORCH_URL=https://download.pytorch.org/whl/
)

REM Install PyTorch
pip install --force-reinstall torch==1.10.0 torchvision==0.11.1 --extra-index-url %TORCH_URL% 

:END
pip install --force-reinstall numpy==1.23

REM Install forked nni to extend to 3d tasks.
pip install --upgrade setuptools pip wheel
git clone https://github.com/audreyeternal/nni.git
cd nni
python setup.py develop --skip-ts 2>NUL
cd ..

