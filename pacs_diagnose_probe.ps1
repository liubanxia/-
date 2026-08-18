$ErrorActionPreference = "SilentlyContinue"

Write-Host "========== Diagnose.exe PACS 只读检查 =========="

$p = Get-CimInstance Win32_Process |
     Where-Object {$_.Name -ieq "Diagnose.exe"} |
     Select-Object -First 1

if (-not $p) {
    Write-Host "没有发现正在运行的 Diagnose.exe"
    exit
}

$pidNow = $p.ProcessId
$exe = $p.ExecutablePath

Write-Host ""
Write-Host "===== 1. PACS主程序 ====="
Write-Host "PID        :" $pidNow
Write-Host "EXE        :" $exe
Write-Host "CommandLine:" $p.CommandLine

if ($exe -and (Test-Path $exe)) {
    $f = Get-Item $exe

    Write-Host "Company    :" $f.VersionInfo.CompanyName
    Write-Host "Product    :" $f.VersionInfo.ProductName
    Write-Host "Description:" $f.VersionInfo.FileDescription
    Write-Host "Version    :" $f.VersionInfo.FileVersion
}

Write-Host ""
Write-Host "===== 2. Diagnose.exe 当前网络连接 ====="

Get-NetTCPConnection -OwningProcess $pidNow |
Select-Object LocalAddress,LocalPort,RemoteAddress,RemotePort,State |
Sort-Object RemoteAddress,RemotePort |
Format-Table -AutoSize

Write-Host ""
Write-Host "===== 3. 父进程 ====="

$parent = Get-CimInstance Win32_Process |
          Where-Object {$_.ProcessId -eq $p.ParentProcessId}

$parent |
Select-Object ProcessId,Name,ExecutablePath,CommandLine |
Format-List

Write-Host ""
Write-Host "===== 4. Diagnose.exe 启动的子进程 ====="

Get-CimInstance Win32_Process |
Where-Object {$_.ParentProcessId -eq $pidNow} |
Select-Object ProcessId,Name,ExecutablePath,CommandLine |
Format-List

Write-Host ""
Write-Host "===== 5. PACS程序目录 ====="

if ($exe) {
    $dir = Split-Path $exe -Parent
    Write-Host $dir

    Write-Host ""
    Write-Host "===== 6. 附近可能的配置文件 ====="

    Get-ChildItem $dir -File -Recurse -Depth 2 |
    Where-Object {
        $_.Extension -match '^\.(ini|cfg|conf|config|xml|json|yaml|yml|txt)$'
    } |
    Select-Object FullName,Length,LastWriteTime |
    Sort-Object FullName |
    Format-Table -AutoSize

    Write-Host ""
    Write-Host "===== 7. 只查PACS/DICOM相关配置项 ====="

    $files = Get-ChildItem $dir -File -Recurse -Depth 2 |
             Where-Object {
                 $_.Extension -match '^\.(ini|cfg|conf|config|xml|json|yaml|yml|txt)$' -and
                 $_.Length -lt 5MB
             }

    foreach ($file in $files) {

        $hits = Select-String `
            -Path $file.FullName `
            -Pattern 'AE.?Title|Called.?AE|Calling.?AE|DICOM|PACS.?Server|RIS.?Server|Report.?Server|Server.?IP|Host|Port|WADO|QIDO|STOW|HL7|StudyInstanceUID|AccessionNumber' `
            -CaseSensitive:$false

        $safe = $hits | Where-Object {
            $_.Line -notmatch 'password|passwd|pwd|secret|token|credential|connection.?string'
        }

        if ($safe) {
            Write-Host ""
            Write-Host "---" $file.FullName "---"

            $safe |
            Select-Object LineNumber,Line |
            Format-Table -Wrap
        }
    }
}

Write-Host ""
Write-Host "========== 检查结束 =========="
