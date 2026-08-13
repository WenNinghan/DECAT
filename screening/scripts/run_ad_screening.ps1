param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$screeningRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $screeningRoot "run_ad_screening_seed242.py"
if (-not (Test-Path -LiteralPath $script)) { throw "AD script not found: $script" }
& $Python $script
if ($LASTEXITCODE -ne 0) { throw "AD screening exited with code $LASTEXITCODE" }

