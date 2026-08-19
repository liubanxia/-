@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title Phoenix 医院电脑离线模型下载

set "WORKBENCH=%~dp0"
for %%I in ("%WORKBENCH%..\..") do set "PROJECT_ROOT=%%~fI"
set "SSD_PY=%PROJECT_ROOT%\02_开发环境\python.exe"
set "LOG=%WORKBENCH%医院模型下载.log"

if exist "%SSD_PY%" (
    set "PY=%SSD_PY%"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PY=py -3"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PY=python"
        ) else (
            echo [错误] 找不到 Python。
            echo 预期位置：%SSD_PY%
            pause
            exit /b 1
        )
    )
)

set "HF_HUB_DOWNLOAD_TIMEOUT=120"
set "HF_HUB_ETAG_TIMEOUT=30"
set "PHOENIX_HF_MIRRORS=https://hf-mirror.com"

echo ============================================================
echo Phoenix 医院电脑模型下载
echo 不需要 GPT，不需要 VPN。自动线路：
echo   1. ModelScope
echo   2. hf-mirror.com 社区镜像
echo   3. Hugging Face 官方直连
echo 下载失败会自动切换线路并保留已下载文件。
echo ============================================================
echo.

call :ensure_downloader
if errorlevel 1 goto :fatal

:menu
echo.
echo [1] 医院推荐：Marian + Embedding + Qwen3.5-2B 快速生成
echo [2] 完整翻译：Marian + NLLB + Qwen3.5-2B
echo [3] 轻量翻译：只下载 Marian
echo [4] 知识整理：Embedding + Qwen3.5-2B + Reranker
echo [5] 全部模型（含2B快速 + 4B深度）
echo [6] 仅 Qwen3.5-2B 快速生成
echo [7] 仅 Qwen3.5-4B 深度质量
echo [8] 仅 NLLB-600M
echo [9] 仅 Embedding-0.6B
echo [10] 查看自动下载线路
echo [0] 退出
echo.
set /p "CHOICE=请输入编号："

if "%CHOICE%"=="1" set "TARGET=hospital_recommended"& goto :download
if "%CHOICE%"=="2" set "TARGET=translation"& goto :download
if "%CHOICE%"=="3" set "TARGET=translation_light"& goto :download
if "%CHOICE%"=="4" set "TARGET=knowledge"& goto :download
if "%CHOICE%"=="5" set "TARGET=all"& goto :download
if "%CHOICE%"=="6" set "TARGET=generator_fast"& goto :download
if "%CHOICE%"=="7" set "TARGET=generator"& goto :download
if "%CHOICE%"=="8" set "TARGET=translation_backup"& goto :download
if "%CHOICE%"=="9" set "TARGET=embedding"& goto :download
if "%CHOICE%"=="10" goto :routes
if "%CHOICE%"=="0" exit /b 0

echo 输入无效。
goto :menu

:download
echo.
echo [%date% %time%] START %TARGET% >> "%LOG%"
echo 开始下载：%TARGET%
echo 下载过程会直接显示在本窗口；每个模型同时写入 PHOENIX_DOWNLOAD_STATUS.json。
call %PY% "%WORKBENCH%model_download.py" %TARGET% --source auto --retries 2 --retry-delay 3
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (
    echo.
    echo [完成] 模型已经下载并通过基础完整性检查。
) else (
    echo.
    echo [未全部完成] 已下载的部分会保留，下次再运行同一项目会继续尝试。
    echo 下载状态保存在各模型目录的 PHOENIX_DOWNLOAD_STATUS.json。
)
echo [%date% %time%] END %TARGET% rc=%RC% >> "%LOG%"
pause
goto :menu

:routes
call %PY% "%WORKBENCH%model_download.py" translation_light --source auto --list-routes
pause
goto :menu

:ensure_downloader
call %PY% -c "import huggingface_hub, modelscope" >nul 2>nul
if not errorlevel 1 exit /b 0

echo 下载器依赖缺失，开始自动尝试 Python 软件源……
call :pip_try "https://pypi.tuna.tsinghua.edu.cn/simple"
if not errorlevel 1 exit /b 0
call :pip_try "https://mirrors.aliyun.com/pypi/simple/"
if not errorlevel 1 exit /b 0
call :pip_try "https://pypi.org/simple"
if not errorlevel 1 exit /b 0
exit /b 1

:pip_try
set "INDEX=%~1"
echo 尝试：%INDEX%
call %PY% -m pip install --disable-pip-version-check --timeout 60 --retries 2 -i "%INDEX%" "huggingface_hub>=0.30" "modelscope>=1.26"
if errorlevel 1 exit /b 1
call %PY% -c "import huggingface_hub, modelscope" >nul 2>nul
exit /b %ERRORLEVEL%

:fatal
echo.
echo [失败] 三个 Python 软件源都无法安装下载器依赖。
echo 这通常表示医院网络把外部下载站整体拦截。
echo 此时不要删除任何已下载内容；可改用手机/网吧把模型目录复制到 SSD。
pause
exit /b 1
