@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
set "PY=%ROOT%\02_开发环境\python.exe"

echo ============================================================
echo Phoenix 上线更新 + 快速总检 + 全功能真实验收
echo PROJECT=%ROOT%
echo ============================================================

where git >nul 2>nul
if errorlevel 1 (
    echo UPDATE=SKIPPED
    echo Git不可用，跳过在线更新，继续本机验收。
    goto CHECK
)

pushd "%ROOT%"
for /f "delims=" %%S in ('git status --porcelain 2^>nul') do (
    echo UPDATE=BLOCKED
    echo 检测到本地未提交改动，为避免覆盖，已停止自动pull。
    echo 当前代码仍会继续总检，但结果不代表远端最新版本。
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
echo.
echo ==================== 第一阶段：正式装配链快速总检 ====================
"%PY%" "%HERE%onsite_preflight_final.py"
set "PRE_RC=%ERRORLEVEL%"

if "%PRE_RC%"=="3" (
    echo.
    echo 结论：BLOCKED。基础/输出/架构/正式GUI装配存在硬故障。
    echo 不继续加载大模型，也不要逐个按钮试。
    popd
    pause
    exit /b 3
)

if "%PRE_RC%"=="2" (
    echo.
    echo 结论：DEGRADED。基础功能可运行，但AI能力尚未全部就绪。
    echo 为避免把已知降级误判成全功能通过，本次不执行真实大模型验收。
    popd
    pause
    exit /b 2
)

echo.
echo ==================== 第二阶段：全功能真实验收 ====================
echo 将真实调用：资料检索、智能1、医学翻译、整本PDF输出、
echo 多资料联合整理、多格式导出、GUI正式装配链。
"%PY%" "%HERE%real_acceptance_final.py"
set "FULL_RC=%ERRORLEVEL%"

popd
echo.
if "%FULL_RC%"=="0" (
    echo ============================================================
    echo PHOENIX_FULL_ACCEPTANCE=PASS
    echo 结论：READY。正式装配链总检和真实功能验收均通过。
    echo ============================================================
    pause
    exit /b 0
) else (
    echo ============================================================
    echo PHOENIX_FULL_ACCEPTANCE=FAIL
    echo 结论：BLOCKED。快速总检虽然通过，但至少一项真实功能调用失败。
    echo 不要逐个按钮试；直接按上方 FAIL 项处理共同根因。
    echo ============================================================
    pause
    exit /b 4
)