# scripts/check.ps1: verification globale avant commit.
#   1. compile le livret (si latexmk est disponible)
#   2. lance pytest
#   3. execute chaque notebook marimo numerote comme un script
#
# Usage (depuis la racine du toolchain) :
#   pwsh scripts/check.ps1            tout
#   pwsh scripts/check.ps1 -NoLatex   sans le livret
#   pwsh scripts/check.ps1 -Only 03   seulement les notebooks dont le nom contient "03"

param(
    [switch]$NoLatex,
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$livret = Join-Path (Split-Path -Parent $root) "ece-calcul-embarque-ESP32-2026\redaction-tp"
$failures = @()

function Step($name) { Write-Host "`n=== $name ===" -ForegroundColor Cyan }

# --- 1. Livret ---
if (-not $NoLatex) {
    Step "Livret LaTeX"
    $miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
    if (Test-Path $miktex) { $env:PATH = "$miktex;$env:PATH" }
    # latexmk (MiKTeX) a besoin de perl : a defaut d'un perl installe, celui fourni avec Git
    $gitPerl = Join-Path $env:ProgramFiles "Git\usr\bin"
    if (-not (Get-Command perl -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $gitPerl "perl.exe"))) {
        $env:PATH = "$env:PATH;$gitPerl"
    }
    if (Get-Command latexmk -ErrorAction SilentlyContinue) {
        Push-Location $livret
        latexmk -pdf -interaction=nonstopmode main.tex *> latexmk.out
        $code = $LASTEXITCODE
        $undef = Select-String -Path main.log -Pattern "undefined" -SimpleMatch -ErrorAction SilentlyContinue
        Pop-Location
        if ($code -ne 0) { $failures += "livret (latexmk exit $code, voir redaction-tp/main.log)" }
        elseif ($undef) { $failures += "livret (references ou citations non definies)" }
        else { Write-Host "OK: main.pdf" -ForegroundColor Green }
    } else {
        Write-Host "latexmk introuvable, livret ignore" -ForegroundColor Yellow
    }
}

# --- 2. pytest ---
Step "pytest"
Push-Location $root
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { $failures += "pytest" }

# --- 3. Notebooks ---
Step "Notebooks"
$env:MPLBACKEND = "Agg"
$env:PYTHONIOENCODING = "utf-8"
# -Filter ne gere pas les plages [0-9] : selection par expression reguliere
$notebooks = Get-ChildItem -Path (Join-Path $root "tps") -Recurse -File -Filter "*.py" |
    Where-Object { $_.Name -match '^\d' -and $_.Name -like "*$Only*" } | Sort-Object Name
foreach ($nb in $notebooks) {
    $t0 = Get-Date
    & $py $nb.FullName *> "$env:TEMP\edgekit-nb.log"
    $code = $LASTEXITCODE
    $dt = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($code -eq 0) {
        Write-Host ("OK   {0,-40} {1,6} s" -f $nb.Name, $dt) -ForegroundColor Green
    } else {
        Write-Host ("FAIL {0,-40} {1,6} s" -f $nb.Name, $dt) -ForegroundColor Red
        Get-Content "$env:TEMP\edgekit-nb.log" -Tail 15
        $failures += "notebook $($nb.Name)"
    }
}
Pop-Location

# --- Bilan ---
Step "Bilan"
if ($failures.Count -eq 0) {
    Write-Host "Tout passe." -ForegroundColor Green
    exit 0
}
$failures | ForEach-Object { Write-Host "ECHEC: $_" -ForegroundColor Red }
exit 1
