$ErrorActionPreference = "Stop"

$caseRoot = "E:\Geysering\tests\test_02_cong2017\cases\BH3_Dr26_H066_L061\openfoam\2d"
$resultRoot = Join-Path $caseRoot "results"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$arguments = @(
    "-d", "Ubuntu",
    "--cd", "/mnt/e/Geysering/tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d",
    "--", "python3", "finalize_when_done.py"
)
$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $resultRoot "finalizer.stdout.log") `
    -RedirectStandardError (Join-Path $resultRoot "finalizer.stderr.log") `
    -PassThru

[ordered]@{
    schema_version = 1
    case = "BH3_Dr26_H066_L061"
    windows_pid = $process.Id
    started_at = (Get-Date).ToString("o")
    state = "STARTED"
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resultRoot "finalizer_launch_record.json") -Encoding utf8
