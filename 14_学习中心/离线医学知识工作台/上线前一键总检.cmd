@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
set "PY=%ROOT%\02_开发环境\python.exe"

echo ============================================================
echo Phoenix 上线前一键总检
echo PROJECT=%ROOT%
echo ============================================================

if not exist "%PY%" (
    echo PHOENIX_ONSITE_PREFLIGHT=BLOCKED
    echo FAIL  项目Python不存在：%PY%
    echo 处理：确认SSD工程完整，并从项目根目录运行本文件。
    pause
    exit /b 3
)

pushd "%HERE%"
"%PY%" "%HERE%onsite_preflight.py"
set "RC=%ERRORLEVEL%"
popd

echo.
if "%RC%"=="0" (
    echo 结论：READY，可以进入工作台。
) else if "%RC%"=="2" (
    echo 结论：DEGRADED，基础功能可启动，但上面列出的AI能力需要先修。
) else (
    echo 结论：BLOCKED，不要逐个功能试，先按上面的共同根因修复。
)
echo.
pause
exit /b %RC%
