@echo off
setlocal

cd /d "%~dp0"

set "ROOT=%~dp0"
set "PY=%ROOT%04_AI模型\工程工作区\phoenix_distill_env\Scripts\pythonw.exe"
set "PYC=%ROOT%04_AI模型\工程工作区\phoenix_distill_env\Scripts\python.exe"
set "APP=%ROOT%Phoenix_GUI.py"

if not exist "%APP%" (
    powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('找不到 Phoenix_GUI.py','Phoenix 启动失败')"
    exit /b 1
)

if exist "%PY%" (
    start "" "%PY%" "%APP%"
    exit /b 0
)

if exist "%PYC%" (
    start "" "%PYC%" "%APP%"
    exit /b 0
)

powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('找不到 Phoenix Python 环境','Phoenix 启动失败')"

exit /b 1
