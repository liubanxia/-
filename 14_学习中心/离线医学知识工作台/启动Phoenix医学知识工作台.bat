@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Phoenix 医学知识工作台

for %%I in ("%~dp0..\..") do set "PHOENIX_PROJECT_ROOT=%%~fI"
set "PY=%PHOENIX_PROJECT_ROOT%\02_开发环境\python.exe"

if not exist "%PY%" (
    echo.
    echo Phoenix 未找到随SSD携带的 Python:
    echo %PY%
    echo.
    echo 请确认 Project Phoenix SSD 目录完整。
    pause
    exit /b 2
)

set "PHOENIX_KNOWLEDGE_ACCELERATOR=auto"

"%PY%" "%~dp0runtime_preflight.py" --repair
if errorlevel 1 (
    echo.
    echo Phoenix 启动自检未通过。上方已给出具体原因。
    pause
    exit /b 2
)

"%PY%" "%~dp0app.py"
if errorlevel 1 (
    echo.
    echo Phoenix 工作台异常退出，错误码 %errorlevel%。
    pause
)

endlocal
