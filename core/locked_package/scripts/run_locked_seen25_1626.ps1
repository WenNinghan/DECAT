# Launch canonical DECAT training or validate the package without training.
param(
    [switch]$VerifyOnly,
    [string]$OutputRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$packageRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "_repro_launch.py"
$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $Python -PathType Leaf) {
    $pythonExecutable = (Resolve-Path -LiteralPath $Python).Path
} else {
    throw "Python interpreter not found: $Python"
}

if ($VerifyOnly) {
    & $pythonExecutable -B $launcher --verify-only
    if ($LASTEXITCODE -ne 0) { throw "Package verification failed with exit code $LASTEXITCODE" }
    exit 0
}

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $packageRoot "outputs/repro_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$stdout = Join-Path $OutputRoot "stdout.log"
$stderr = Join-Path $OutputRoot "stderr.log"

& $pythonExecutable -u $launcher --output-root $OutputRoot 1> $stdout 2> $stderr
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath $stderr -Tail 50 -ErrorAction SilentlyContinue
    throw "Training failed with exit code $LASTEXITCODE (see $stderr)"
}
Write-Host "Training completed. Outputs: $OutputRoot"
