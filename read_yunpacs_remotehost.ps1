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

$projectRoot = Get-PhoenixProjectRoot
$db = Join-Path $projectRoot "pacs_structure_copy.mdb"

if (-not (Test-Path -LiteralPath $db)) {
    throw "Inspection MDB copy not found: $db"
}

$conn = New-Object System.Data.OleDb.OleDbConnection
$conn.ConnectionString = "Provider=Microsoft.Jet.OLEDB.4.0;Data Source=$db;Mode=Read;"
$conn.Open()

$tables = @(
    "RemoteHost",
    "StoreInfo"
)

foreach ($table in $tables) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "TABLE: $table"
    Write-Host "========================================"

    try {
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = "SELECT * FROM [$table]"

        $reader = $cmd.ExecuteReader()
        $rowNo = 0

        while ($reader.Read()) {
            $rowNo++
            Write-Host ""
            Write-Host "ROW:" $rowNo

            for ($i = 0; $i -lt $reader.FieldCount; $i++) {
                $name = $reader.GetName($i)

                if ($name -match "password|passwd|pwd|secret|token|credential|connectionstring") {
                    Write-Host ($name + " = [REDACTED]")
                    continue
                }

                $value = $reader.GetValue($i)

                if ($value -eq [DBNull]::Value) {
                    $value = ""
                }

                Write-Host ($name + " = " + $value)
            }
        }

        if ($rowNo -eq 0) {
            Write-Host "NO ROWS"
        }

        $reader.Close()
    }
    catch {
        Write-Host "ERROR:" $_.Exception.Message
    }
}

$conn.Close()

Write-Host ""
Write-Host "========== FINISHED =========="
Write-Host "READ-ONLY COPY USED."
