# Start the local Apertus server.
#
#   powershell scripts/start_apertus.ps1
#   powershell scripts/start_apertus.ps1 -Model 8b
#
# llama.cpp is not installed system-wide on this box, so the binaries live in a
# temp directory and are fetched on first run. Everything else is local: after
# the model download this runs with no network access.

param(
    [ValidateSet("4b", "8b")]
    [string]$Model = "4b",
    [int]$Port = 8090,
    [int]$Threads = 0,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModelDir = Join-Path $RepoRoot "models"
$BinDir = Join-Path $env:LOCALAPPDATA "opencode\llamacpp"

if ($Threads -le 0) {
    # Leave one core for the OS and the API process.
    $Cores = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
    $Threads = [Math]::Max(2, $Cores - 2)
}

$Models = @{
    "4b" = @{ File = "Apertus-v1.1-4B-Instruct-Q4_K_M.gguf"; Context = 4096; Layers = 99 }
    "8b" = @{ File = "Apertus-8B-Instruct-2509-Q4_K_M.gguf"; Context = 8192; Layers = 99 }
}
$Chosen = $Models[$Model]
$ModelPath = Join-Path $ModelDir $Chosen.File

# ---- model ------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model not found: $ModelPath`nDownload it first (see README.md)."
}

# ---- binaries ---------------------------------------------------------------
$Server = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path -LiteralPath $Server)) {
    throw "llama-server.exe not found at $Server`nRun scripts/fetch_llamacpp.ps1 first."
}

# ---- already running --------------------------------------------------------
$Existing = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "llama-server already running (pid $($Existing.ProcessId -join ', ')) on port $Port."
    Write-Host "Stop it first with scripts/stop_apertus.ps1 if you want to switch models."
    exit 0
}

$Args = @(
    "-m", $ModelPath,
    "-ngl", $Chosen.Layers,          # full Vulkan offload to the iGPU
    "--host", "127.0.0.1",
    "--port", $Port,
    "-c", $Chosen.Context,
    "-t", $Threads,
    "--jinja",                       # Apertus uses an Llama-3-style chat template
    "--no-webui",
    "--alias", "apertus-$Model"
)

Write-Host "model   : $($Chosen.File)"
Write-Host "context : $($Chosen.Context)"
Write-Host "threads : $Threads"
Write-Host "url     : http://127.0.0.1:$Port/v1"
Write-Host ""

if ($Foreground) {
    & $Server @Args
    exit $LASTEXITCODE
}

# Detached: we want this to survive the shell that launched it.
$StartParams = @{
    FilePath               = $Server
    ArgumentList           = $Args
    WorkingDirectory       = $BinDir
    WindowStyle            = "Hidden"
    RedirectStandardOutput = (Join-Path $BinDir "server-$Model.log")
    RedirectStandardError  = (Join-Path $BinDir "server-$Model.err.log")
}
try {
    Start-Process @StartParams -PassThru | Out-Null
} catch {
    # Redirecting both streams at once is unsupported for -NoNewWindow on some
    # PowerShell builds; fall back to a plain hidden window.
    Start-Process -FilePath $Server -ArgumentList $Args -WorkingDirectory $BinDir `
        -WindowStyle Hidden | Out-Null
}

# Wait for the model to finish loading. On CPU/Vulkan this takes 10-40s.
$Ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($r.status -eq "ok") { $Ready = $true; break }
    } catch {
        # still loading
    }
}

if ($Ready) {
    Write-Host "Apertus $Model is ready at http://127.0.0.1:$Port/v1"
} else {
    Write-Warning "Server did not report healthy within 120s."
    Write-Warning "Check logs: $BinDir\server-$Model.log"
    exit 1
}