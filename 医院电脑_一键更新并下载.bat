@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title Phoenix 医院电脑一键更新并下载

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "BRANCH=refactor/minimal-phoenix"
set "DOWNLOADER=%PROJECT_ROOT%\14_学习中心\离线医学知识工作台\医院电脑_一键下载模型.bat"

echo ============================================================
echo Phoenix 医院电脑一键更新并下载
echo 1. 更新昨晚/今天全部 Phoenix 代码
echo 2. 自动进入无VPN多线路模型下载菜单
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 Git，无法更新 Phoenix。
    pause
    exit /b 1
)

rem 医院/网吧外接SSD在不同电脑可能被Git判定为不同所有者。
git config --global --add safe.directory "%PROJECT_ROOT%" >nul 2>nul

pushd "%PROJECT_ROOT%"

echo [1/3] 检查本地工作区……
for /f "delims=" %%L in ('git status --porcelain 2^>nul') do (
    echo [停止] 检测到尚未提交的本地代码修改，为避免覆盖文件，本次不自动更新。
    echo 请保留当前窗口并记录下面内容：
    git status --short
    popd
    pause
    exit /b 2
)

echo [2/3] 获取并切换到 %BRANCH% ……
git fetch origin "%BRANCH%"
if errorlevel 1 goto :git_failed

git checkout "%BRANCH%"
if errorlevel 1 goto :git_failed

git pull --ff-only origin "%BRANCH%"
if errorlevel 1 goto :git_failed

echo.
echo [3/3] Phoenix 代码更新完成。
git log -1 --oneline
popd

if not exist "%DOWNLOADER%" (
    echo.
    echo [错误] 更新完成，但没有找到模型下载脚本：
    echo %DOWNLOADER%
    pause
    exit /b 3
)

echo.
echo 即将进入模型下载菜单……
call "%DOWNLOADER%"
exit /b %ERRORLEVEL%

:git_failed
echo.
echo [失败] GitHub 代码更新没有完成。
echo 本地文件没有被强制覆盖；可稍后重新双击本脚本继续。
popd
pause
exit /b 4
