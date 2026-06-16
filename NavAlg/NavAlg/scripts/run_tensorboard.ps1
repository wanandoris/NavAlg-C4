param(
    [int]$Port = 6006
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $RepoRoot "runs"
Write-Host "tensorboard --logdir `"$LogDir`" --port $Port"
tensorboard --logdir "$LogDir" --port $Port
