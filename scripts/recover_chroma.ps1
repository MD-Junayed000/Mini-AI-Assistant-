# recover_chroma.ps1
# One-shot recovery for a corrupted ChromaDB persistent directory.
# See docs/runbook.md for the full story.

[CmdletBinding()]
param(
    [string]$ChromaDir = ".chroma",
    [string]$Pattern   = "uvicorn"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Write-Step($msg) { Write-Host "[recover] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[recover] WARN: $msg" -ForegroundColor Yellow }

# 1. Stop uvicorn so chromadb releases its mmap-ed files.
Write-Step "stopping any running $Pattern processes"
$procs = @(Get-Process -Name python -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$Pattern*" })
if ($procs.Count -gt 0) {
    foreach ($p in $procs) {
        Write-Step ("  -> stopping PID " + $p.Id)
        Stop-Process -Id $p.Id -Force
    }
    Start-Sleep -Seconds 1
} else {
    Write-Step "  (none found)"
}

# 2. Move-aside the corrupt directory.
if (-not (Test-Path $ChromaDir)) {
    Write-Warn "$ChromaDir does not exist - nothing to move. Will be recreated on next ingest."
    exit 0
}

$stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
$backup = "$ChromaDir.bak-$stamp"
Write-Step "moving $ChromaDir -> $backup"

try {
    Move-Item -Path $ChromaDir -Destination $backup -Force
} catch {
    Write-Warn "Move-Item failed: $_"
    Write-Warn "If files are still locked, stop the offending process manually and retry."
    exit 1
}

# 3. Report so the operator can decide whether to re-ingest.
$size = (Get-ChildItem $backup -Recurse -Force -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
$count = (Get-ChildItem $backup -Recurse -File -ErrorAction SilentlyContinue).Count
$sizeMb = "{0:N1}" -f ($size / 1MB)
Write-Step ("backup ready: {0:N0} files, {1} MB" -f $count, $sizeMb)
Write-Step "next /ingest call will recreate the collection from data/"
$hint = "to inspect the corrupt files later:  explorer $backup"
Write-Step $hint