param(
    [string]$Python = "python",
    [int]$Port = 8501,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$deploymentRoot = Split-Path -Parent $PSScriptRoot
$app = Join-Path $deploymentRoot "streamlit_app.py"
if (-not (Test-Path -LiteralPath $app)) { throw "Streamlit entrypoint not found: $app" }

$args = @("-m", "streamlit", "run", $app, "--server.port", [string]$Port)
if ($Headless) { $args += @("--server.headless", "true") }
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }

