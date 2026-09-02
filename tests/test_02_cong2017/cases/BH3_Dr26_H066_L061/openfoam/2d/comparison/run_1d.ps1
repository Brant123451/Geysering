$ErrorActionPreference = "Stop"

$comparisonRoot = "E:\Geysering\tests\test_02_cong2017\cases\BH3_Dr26_H066_L061\openfoam\2d\comparison"
$outputRoot = Join-Path $comparisonRoot "model_1d"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$arguments = @(
    "-d", "Ubuntu",
    "--cd", "/mnt/e/Geysering/tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d",
    "--", "python3", "comparison/run_paper_layout_1d.py"
)

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputRoot "run.stdout.log") `
    -RedirectStandardError (Join-Path $outputRoot "run.stderr.log") `
    -PassThru

$record = [ordered]@{
    schema_version = 1
    case = "BH3_Dr26_H066_L061"
    geometry = "paper_Fig_1b"
    windows_pid = $process.Id
    started_at = (Get-Date).ToString("o")
    state = "STARTED"
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputRoot "launch_record.json") -Encoding utf8
$record | ConvertTo-Json
