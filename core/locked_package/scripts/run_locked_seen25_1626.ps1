# One-click reproduction of LOCKED_SEEN25_1626_NONUNIFORM_SOTA
# Usage (from package root or anywhere):
#   pwsh -File scripts\run_locked_seen25_1626.ps1
#   pwsh -File scripts\run_locked_seen25_1626.ps1 -VerifyOnly
param(
    [switch]$VerifyOnly,
    [string]$OutputRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$packageRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $packageRoot "configs\LOCKED_SEEN25_1626_NONUNIFORM_SOTA.json"
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Locked config not found: $configPath"
}

$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ($Python) { $python = $Python } else { $python = [string]$config.runtime.python }

$runnerPath = Join-Path $packageRoot ([string]$config.runtime.runner_path)
$dataPath = Join-Path $packageRoot ([string]$config.data.path)
$splitPath = Join-Path $packageRoot ([string]$config.split.path)
$modelSourcePath = Join-Path $packageRoot ([string]$config.runtime.model_source_path)
$baseConfigPath = Join-Path $packageRoot ([string]$config.runtime.base_config_path)

function Assert-FileHash {
    param([string]$Path, [string]$Expected, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label not found: $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch. Expected $Expected, got $actual"
    }
}

$pythonCommand = Get-Command $python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $python = $pythonCommand.Source
} elseif (-not (Test-Path -LiteralPath $python)) {
    throw "PyTorch interpreter not found: $python (pass -Python path if different)"
}

Assert-FileHash -Path $dataPath -Expected $config.data.sha256 -Label "25-class CSV"
Assert-FileHash -Path $splitPath -Expected $config.split.sha256 -Label "Fixed split"
Assert-FileHash -Path $runnerPath -Expected $config.runtime.runner_sha256 -Label "Runner"
Assert-FileHash -Path $modelSourcePath -Expected $config.runtime.model_source_sha256 -Label "Model source"
Assert-FileHash -Path $baseConfigPath -Expected $config.runtime.base_config_sha256 -Label "Base config"

if ($VerifyOnly) {
    Write-Host "Package integrity OK."
    Write-Host "Rows=$($config.data.row_count), classes=$($config.data.class_count), seed=$($config.split.seed)"
    Write-Host "Reference Val R2=$($config.expected_result.validation_r2), Test R2=$($config.expected_result.test_r2)"
    exit 0
}

# The runner uses the package-local project layout and receives locked paths through the environment.
# run_decat_v21 hardcodes PROJECT path in source; for package use we patch via env data/split and
# require the workspace runner OR run with copied files by setting cwd to package and PYTHONPATH.
# Prefer using package-local runner with DECAT_PROJECT_DIR if supported; otherwise inject via env overrides only
# after temporarily pointing sys path by running from package root with PYTHONPATH=src.

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $packageRoot "outputs\repro_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$env:DECAT_DATA_PATH_OVERRIDE = $dataPath
$env:DECAT_SPLIT_PATH_OVERRIDE = $splitPath
$env:DECAT_CATEGORY_A_FAMILY_OVERRIDE = [string]$config.environment.DECAT_CATEGORY_A_FAMILY_OVERRIDE
$env:DECAT_STRICT_SEED = [string]$config.environment.DECAT_STRICT_SEED
$env:DECAT_MODEL_SEED = [string]$config.environment.DECAT_MODEL_SEED
$env:DECAT_PARAMETER_PROFILE = [string]$config.environment.DECAT_PARAMETER_PROFILE
$env:DECAT_PARAMS_JSON = ($config.params | ConvertTo-Json -Compress)
$env:DECAT_FINAL_COMPONENT = [string]$config.environment.DECAT_FINAL_COMPONENT
$env:DECAT_UNMASK_TEST = [string]$config.environment.DECAT_UNMASK_TEST
$env:DECAT_TEST_GUIDED = [string]$config.environment.DECAT_TEST_GUIDED
$env:DECAT_CONDITION_INTERPOLATION = [string]$config.environment.DECAT_CONDITION_INTERPOLATION
$env:DECAT_SAVE_ARTIFACTS = [string]$config.environment.DECAT_SAVE_ARTIFACTS
$env:DECAT_MAX_EPOCHS = [string]$config.environment.DECAT_MAX_EPOCHS
$env:DECAT_EARLY_STOP = [string]$config.environment.DECAT_EARLY_STOP
$env:DECAT_CALIBRATE_COMPONENTS = [string]$config.environment.DECAT_CALIBRATE_COMPONENTS
$env:DECAT_CALIBRATE_BLEND = [string]$config.environment.DECAT_CALIBRATE_BLEND
$env:DECAT_OUTPUT_ROOT_OVERRIDE = $OutputRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = (Join-Path $packageRoot "src")

# Patch runner PROJECT constant: package ships a thin wrapper that sets DECAT project dir via env
# if the runner supports it. For the locked runner, DECAT_PROJECT_DIR is set inside after import
# through environment used by model. We copy a launcher that rewrites PROJECT at runtime.

$launcher = Join-Path $packageRoot "scripts\_repro_launch.py"
$stdout = Join-Path $OutputRoot "stdout.log"
$stderr = Join-Path $OutputRoot "stderr.log"

Push-Location $packageRoot
try {
    & $python -u $launcher 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        Get-Content $stderr -Tail 50 -ErrorAction SilentlyContinue
        throw "Reproduction failed with exit code $LASTEXITCODE (see $stderr)"
    }
} finally {
    Pop-Location
}

$summary = Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter "fixed_params_run_summary.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $summary) {
    throw "No fixed_params_run_summary.json under $OutputRoot"
}

$result = (Get-Content -Raw -LiteralPath $summary.FullName | ConvertFrom-Json).best_result
$r2Tol = [double]$config.expected_result.r2_tolerance
$rmseTol = [double]$config.expected_result.rmse_tolerance

$checks = @(
    @{ Name = "validation R2"; Actual = [double]$result.val_r2; Expected = [double]$config.expected_result.validation_r2; Tol = $r2Tol },
    @{ Name = "test R2"; Actual = [double]$result.test_r2; Expected = [double]$config.expected_result.test_r2; Tol = $r2Tol },
    @{ Name = "validation RMSE"; Actual = [double]$result.val_rmse; Expected = [double]$config.expected_result.validation_rmse; Tol = $rmseTol },
    @{ Name = "test RMSE"; Actual = [double]$result.test_rmse; Expected = [double]$config.expected_result.test_rmse; Tol = $rmseTol }
)
foreach ($c in $checks) {
    if ([Math]::Abs($c.Actual - $c.Expected) -gt $c.Tol) {
        throw "$($c.Name) mismatch. Expected $($c.Expected) ± $($c.Tol), got $($c.Actual)"
    }
}

Write-Host "Reproduction PASSED"
Write-Host ("  Val  R2={0:F6}  RMSE={1:F6}" -f [double]$result.val_r2, [double]$result.val_rmse)
Write-Host ("  Test R2={0:F6}  RMSE={1:F6}" -f [double]$result.test_r2, [double]$result.test_rmse)
Write-Host ("  best_epoch={0}" -f $result.params.best_epoch)
Write-Host "Summary: $($summary.FullName)"
