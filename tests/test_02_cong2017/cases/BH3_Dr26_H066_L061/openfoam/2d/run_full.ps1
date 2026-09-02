$ErrorActionPreference = "Stop"

$caseRoot = "E:\Geysering\tests\test_02_cong2017\cases\BH3_Dr26_H066_L061\openfoam\2d"
$resultRoot = Join-Path $caseRoot "results"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$arguments = @(
    "-d", "Ubuntu",
    "--cd", "/mnt/e/Geysering/tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d",
    "--", "env", "OPENFOAM_NP=3", "bash", "./Allrun", "solve"
)

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList $arguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $resultRoot "full_run_driver.stdout.log") `
    -RedirectStandardError (Join-Path $resultRoot "full_run_driver.stderr.log") `
    -PassThru

$record = [ordered]@{
    schema_version = 1
    case = "BH3_Dr26_H066_L061"
    paper_run = "B-H3"
    windows_pid = $process.Id
    openfoam_np = 3
    started_at = (Get-Date).ToString("o")
    state = "STARTED"
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resultRoot "launch_record.json") -Encoding utf8
$record | ConvertTo-Json
