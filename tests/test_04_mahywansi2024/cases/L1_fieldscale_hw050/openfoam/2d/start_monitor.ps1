$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$solverPid = (Get-Content -LiteralPath (Join-Path $root 'run.pid') -Raw).Trim()
$runtimeRoot = (Get-Content -LiteralPath (Join-Path $root 'runtime_path.txt') -Raw).Trim()
$monitorWindowsPath = Join-Path $root 'monitor_and_sync.sh'
$monitorWslPath = (wsl.exe -e wslpath -a $monitorWindowsPath).Trim()

$arguments = @(
    '-e',
    'bash',
    $monitorWslPath,
    $solverPid,
    $runtimeRoot
)

$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments `
    -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $root 'monitor.windows.pid')
Write-Output "Started status monitor: Windows PID $($process.Id)"
