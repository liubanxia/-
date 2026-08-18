$ErrorActionPreference = "Stop"

function Get-PhoenixProjectRoot {
    if ($env:PHOENIX_PROJECT_ROOT -and (Test-Path -LiteralPath $env:PHOENIX_PROJECT_ROOT)) {
        return (Resolve-Path -LiteralPath $env:PHOENIX_PROJECT_ROOT).Path
    }

    foreach ($candidate in @("G:\project_phoenix", "D:\project_phoenix")) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Phoenix project root not found."
}

Write-Host "========== YUNPACS MDB STRUCTURE CHECK =========="

$db = Get-ChildItem "D:\YUNPACS" -Recurse -File -Filter "pacs.mdb" -ErrorAction SilentlyContinue |
      Select-Object -First 1

if (-not $db) {
    Write-Host "ERROR: pacs.mdb not found"
    exit 1
}

Write-Host ""
Write-Host "SOURCE:"
Write-Host $db.FullName
Write-Host "SIZE:" $db.Length
Write-Host "MODIFIED:" $db.LastWriteTime

$projectRoot = Get-PhoenixProjectRoot
$copy = Join-Path $projectRoot "pacs_structure_copy.mdb"

Write-Host ""
Write-Host "Creating inspection copy:"
Write-Host $copy

Copy-Item -LiteralPath $db.FullName -Destination $copy -Force

$providers = @(
    "Microsoft.Jet.OLEDB.4.0",
    "Microsoft.ACE.OLEDB.12.0",
    "Microsoft.ACE.OLEDB.16.0"
)

$conn = $null
$used = $null

foreach ($provider in $providers) {
    try {
        Write-Host ""
        Write-Host "Trying provider:" $provider

        $c = New-Object System.Data.OleDb.OleDbConnection
        $c.ConnectionString = "Provider=$provider;Data Source=$copy;Mode=Read;"
        $c.Open()

        $conn = $c
        $used = $provider

        Write-Host "CONNECTED:" $provider
        break
    }
    catch {
        Write-Host "FAILED:" $provider
        Write-Host $_.Exception.Message
    }
}

if (-not $conn) {
    Write-Host ""
    Write-Host "ERROR: No usable MDB OLEDB provider"
    exit 2
}

Write-Host ""
Write-Host "========== TABLES =========="

$tables = $conn.GetSchema("Tables") |
Where-Object {
    $_.TABLE_TYPE -eq "TABLE" -and
    $_.TABLE_NAME -notlike "MSys*"
} |
Sort-Object TABLE_NAME

$tables |
Select-Object TABLE_NAME |
Format-Table -AutoSize

Write-Host ""
Write-Host "========== COLUMNS =========="

foreach ($t in $tables) {
    $name = [string]$t.TABLE_NAME

    Write-Host ""
    Write-Host "----- TABLE:" $name "-----"

    try {
        $restrictions = New-Object string[] 4
        $restrictions[0] = $null
        $restrictions[1] = $null
        $restrictions[2] = $name
        $restrictions[3] = $null

        $cols = $conn.GetSchema("Columns", $restrictions)

        $cols |
        Sort-Object ORDINAL_POSITION |
        Select-Object COLUMN_NAME,DATA_TYPE,CHARACTER_MAXIMUM_LENGTH |
        Format-Table -AutoSize
    }
    catch {
        Write-Host "COLUMN READ ERROR:" $_.Exception.Message
    }
}

Write-Host ""
Write-Host "========== INTERESTING TABLES =========="

$tables |
Where-Object {
    $_.TABLE_NAME -match "config|setting|server|dicom|pacs|ris|report|study|exam|image|work|patient|doctor"
} |
Select-Object TABLE_NAME |
Format-Table -AutoSize

$conn.Close()

Write-Host ""
Write-Host "========== FINISHED =========="
Write-Host "No data rows were read."
Write-Host "Original MDB was not modified."
