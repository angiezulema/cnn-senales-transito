# HU07 - Configuracion del entorno en Windows
# Uso:
#   powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
#   powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1 -InstallPython

param(
    [switch]$InstallPython,
    [switch]$Force,
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$HU07Root = Split-Path -Parent $PSScriptRoot
Set-Location $HU07Root

Write-Host "HU07 - entorno en $HU07Root" -ForegroundColor Cyan

function Find-Python313 {
    $candidates = @()
    try {
        $exe = & py -3.13 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { $candidates += $exe.Trim() }
    } catch { }
    $candidates += @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        "C:\Python313\python.exe",
        "C:\Program Files\Python313\python.exe"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

$python = Find-Python313
if (-not $python) {
    if ($InstallPython) {
        Write-Host "Instalando Python 3.13 con winget..." -ForegroundColor Yellow
        winget install -e --id Python.Python.3.13 --scope user `
            --accept-package-agreements --accept-source-agreements
        $python = Find-Python313
    }
}

if (-not $python) {
    Write-Host "Python 3.13 no encontrado." -ForegroundColor Red
    Write-Host "Instalalo con: winget install -e --id Python.Python.3.13 --scope user"
    Write-Host "O descarga: https://www.python.org/downloads/release/python-3139/"
    Write-Host "Luego vuelve a ejecutar este script."
    exit 1
}

Write-Host "Python 3.13: $python" -ForegroundColor Green

if ($VenvPath) {
    $venv = $VenvPath
} else {
    $venv = Join-Path $HU07Root ".venv"
    # Windows long-path: TensorFlow tiene rutas internas muy largas.
    # Si la ruta del proyecto es larga, usar una ruta corta fuera de el.
    if ($venv.Length -gt 80) {
        $venv = Join-Path $env:LOCALAPPDATA "hu07-venv"
        Write-Host "Ruta del proyecto larga: se usara venv corto en $venv" -ForegroundColor Yellow
    }
}

Set-Content -Path (Join-Path $HU07Root ".venv_path.txt") -Value $venv -Encoding ASCII

if ($Force -and (Test-Path $venv)) {
    Write-Host "Eliminando venv anterior ($venv)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    Write-Host "Creando entorno virtual .venv..." -ForegroundColor Cyan
    & $python -m venv $venv
}

$venvPython = Join-Path $venv "Scripts\python.exe"
Write-Host "Actualizando pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

Write-Host "Instalando dependencias (puede tardar varios minutos)..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $HU07Root "requirements.txt")

Write-Host ""
Write-Host "Entorno listo." -ForegroundColor Green
Write-Host "Python del entorno: $venvPython"
Write-Host "Activa con: & '$venv\Scripts\Activate.ps1'"
Write-Host "Verifica con: python setup\check_env.py --data-root <ruta con archive>"

# Helper de activacion portable
$activateHelper = @"
# Generado por setup_windows.ps1
`$envPath = "$venv"
& "`$envPath\Scripts\Activate.ps1"
Write-Host "HU07 venv activado: `$envPath" -ForegroundColor Green
"@
Set-Content -Path (Join-Path $HU07Root "activate.ps1") -Value $activateHelper -Encoding UTF8
