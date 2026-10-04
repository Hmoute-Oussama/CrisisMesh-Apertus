param(
    [switch]$Quiet
)

$Procs = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue

if (-not $Procs) {
    if (-not $Quiet) { Write-Host "No llama-server process running." }
    exit 0
}

foreach ($p in $Procs) {
    if (-not $Quiet) { Write-Host "Stopping llama-server (pid $($p.ProcessId))" }
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

if (-not $Quiet) {
    Write-Host "Stopped. The model file on disk is untouched."
}