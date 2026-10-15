# Run every experiment against SubQ and a local Laya server, then build the figures.
# Windows PowerShell 5.1+. Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Options: -N 40 -NWording 30 -SkipLaya -SkipSubQ
param(
    [int]$N = 40,
    [int]$NWording = 30,
    [switch]$SkipLaya,
    [switch]$SkipSubQ
)
# Continue: PS 5.1 turns any native stderr into a terminating error under "Stop";
# failures are caught explicitly via $LASTEXITCODE and throw instead.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------- python for the experiments
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "Python not found on PATH" }
Say "experiment python: $py"
& $py -m pip install -q -r requirements.txt matplotlib numpy
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

$backends = @()

# ---------------------------------------------------------------- SubQ
if (-not $SkipSubQ) {
    if (-not $env:SUBQ_API_KEY) { $env:SUBQ_API_KEY = [Environment]::GetEnvironmentVariable("SUBQ_API_KEY", "User") }
    if (-not $env:SUBQ_API_KEY) {
        $env:SUBQ_API_KEY = Read-Host "Paste SubQ API key (stored in your user environment, not in the repo)"
        [Environment]::SetEnvironmentVariable("SUBQ_API_KEY", $env:SUBQ_API_KEY, "User")
    }
    if (-not $env:SUBQ_BUDGET_USD) { $env:SUBQ_BUDGET_USD = "0.50" }
    & $py experiments\exp00_smoke.py
    if ($LASTEXITCODE -ne 0) { throw "SubQ smoke test failed (402 = no credits, 401 = bad key)" }
    $backends += "subq"
}

# ---------------------------------------------------------------- Laya
function Test-LayaUrl($base) {
    $body = '{"model":"english","state":"bids A=3 B=5","questions":{"w":{"type":"choice","instructions":"who bids more?","criteria":{"A":"A","B":"B"}}}}'
    try {
        $r = Invoke-RestMethod -Method Post -Uri "$base/systemone" -ContentType "application/json" -Body $body -TimeoutSec 20
        return [bool]$r.answers
    } catch { return $false }
}

function Find-LayaPython {
    # 1) an env that already has laya-serve installed
    $roots = @("$repo\.venv-laya", $env:USERPROFILE, "D:\") | Where-Object { Test-Path $_ }
    foreach ($r in $roots) {
        $hit = Get-ChildItem -Path $r -Recurse -Depth 5 -Filter "laya-serve.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) {
            $cand = Join-Path $hit.DirectoryName "python.exe"
            if (Test-Path $cand) { return $cand }
        }
    }
    # 2) any installed interpreter that can import laya
    $cands = @()
    $pyl = Get-Command py -ErrorAction SilentlyContinue
    if ($pyl) { $cands += (& py -0p 2>$null) | ForEach-Object { ($_ -split '\s+')[-1] } | Where-Object { $_ -like "*python.exe" } }
    $cands += (Get-Command python -All -ErrorAction SilentlyContinue | ForEach-Object Source)
    foreach ($c in ($cands | Select-Object -Unique)) {
        & $c -c "import laya" 2>$null
        if ($LASTEXITCODE -eq 0) { return $c }
    }
    return $null
}

$layaProc = $null
if (-not $SkipLaya) {
    $layaBase = $null
    foreach ($u in @("http://127.0.0.1:8000/v1", "http://127.0.0.1:8765/v1")) {
        if (Test-LayaUrl $u) { $layaBase = $u; Say "found a running Laya server at $u"; break }
    }
    if (-not $layaBase) {
        $lpy = Find-LayaPython
        if (-not $lpy) {
            Say "no Laya install found; creating .venv-laya (CUDA 12.8 PyTorch + laya[serve], several GB)"
            $base = (Get-Command py -ErrorAction SilentlyContinue)
            if ($base) { & py -3.11 -m venv .venv-laya 2>$null; if ($LASTEXITCODE -ne 0) { & py -3 -m venv .venv-laya } }
            else { & $py -m venv .venv-laya }
            $lpy = "$repo\.venv-laya\Scripts\python.exe"
            & $lpy -m pip install -q --upgrade pip
            & $lpy -m pip install torch --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
        }
        Say "Laya python: $lpy"
        & $lpy -c "import fastapi, uvicorn" 2>$null
        if ($LASTEXITCODE -ne 0) { & $lpy -m pip install -q "laya[serve]"; if ($LASTEXITCODE -ne 0) { throw "laya[serve] install failed" } }
        $device = (& $lpy -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')").Trim()
        Say "Laya device: $device"
        $serve = Join-Path (Split-Path $lpy) "laya-serve.exe"
        if (-not (Test-Path $serve)) { throw "laya-serve.exe not found next to $lpy" }
        $env:LAYA_DEVICE = $device
        $env:LAYA_PRELOAD = "1"
        $env:LAYA_MODELS = "english,typed-decisions"
        $env:LAYA_HOST = "127.0.0.1"
        $env:LAYA_PORT = "8765"
        New-Item -ItemType Directory -Force results | Out-Null
        $layaProc = Start-Process -FilePath $serve -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput "results\laya-serve.out.log" -RedirectStandardError "results\laya-serve.err.log"
        $layaBase = "http://127.0.0.1:8765/v1"
        Say "waiting for laya-serve (first start downloads weights if they are not cached)..."
        $ok = $false
        for ($i = 0; $i -lt 180; $i++) {
            Start-Sleep -Seconds 5
            if ($layaProc.HasExited) { break }
            if (Test-LayaUrl $layaBase) { $ok = $true; break }
        }
        if (-not $ok) {
            Get-Content "results\laya-serve.err.log" -Tail 30 -ErrorAction SilentlyContinue
            throw "laya-serve did not come up; log above (results\laya-serve.err.log)"
        }
    }
    $env:LAYA_BASE_URL = $layaBase
    $backends += "laya", "laya-typed"
}

if ($backends.Count -eq 0) { throw "nothing to run" }
$bargs = @(); foreach ($b in $backends) { $bargs += "--backend", $b }

try {
    Say "exp01 calibration on: $($backends -join ', ')"
    & $py experiments\exp01_calibration.py @bargs --n $N
    if ($LASTEXITCODE -ne 0) { throw "exp01 failed" }
    Say "exp02 wording on: $($backends -join ', ')"
    & $py experiments\exp02_wording.py @bargs --n $NWording
    if ($LASTEXITCODE -ne 0) { throw "exp02 failed" }
    Say "figures"
    & $py analysis\make_figures.py
    if ($LASTEXITCODE -ne 0) { throw "make_figures failed" }
} finally {
    if ($layaProc -and -not $layaProc.HasExited) { Stop-Process -Id $layaProc.Id -Force; Say "stopped laya-serve" }
}
Say "done: results\REPORT.md and figures\"
