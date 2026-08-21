@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
set "PY=%ROOT%\02_开发环境\python.exe"

echo ============================================================
echo Phoenix 上线更新 + 一键总检
echo PROJECT=%ROOT%
echo ============================================================

where git >nul 2>nul
if errorlevel 1 (
    echo UPDATE=SKIPPED
    echo Git不可用，跳过在线更新，继续本机总检。
    goto CHECK
)

pushd "%ROOT%"
for /f "delims=" %%S in ('git status --porcelain 2^>nul') do (
    echo UPDATE=BLOCKED
    echo 检测到本地未提交改动，为避免覆盖，已停止自动pull。
    echo 请先处理Git改动；本机总检仍会继续。
    goto AFTER_PULL
)

echo 正在执行 git pull --ff-only ...
git pull --ff-only
if errorlevel 1 (
    echo UPDATE=FAILED
    echo Git更新失败；不会修改或删除本地文件，继续检查当前版本。
) else (
    echo UPDATE=PASS
)

:AFTER_PULL
popd

:CHECK
if not exist "%PY%" (
    echo PHOENIX_ONSITE_PREFLIGHT=BLOCKED
    echo FAIL  项目Python不存在：%PY%
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
    echo 结论：DEGRADED，先修上面列出的AI能力，再做正式任务。
) else (
    echo 结论：BLOCKED，不要逐个按钮试，先修共同根因。
)
echo.
pause
exit /b %RC%
