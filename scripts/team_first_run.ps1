$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.10, 3.11, or 3.12 is required. Install it and enable Add Python to PATH."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js and npm are required. Install the current Node.js LTS release first."
}

python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12"'

Write-Host "[1/6] Creating the Python environment..."
python -m venv .venv
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install .

Write-Host "[2/6] Installing website dependencies..."
npm --prefix web install

Write-Host "[3/6] Generating the synthetic CEIT demonstration dataset..."
& $Python scripts/generate_ceit_dataset.py --records-per-level 200 --seed 20260831 --output data/ceit_synthetic_students.csv

Write-Host "[4/6] Training the ANN demonstration model..."
& $Python -m student_performance.train --data data/ceit_synthetic_students.csv --dataset-label "Synthetic MTU CEIT Project Dataset" --synthetic-data --task classification --pass-threshold 50 --output artifacts/classifier

Write-Host "[5/6] Creating your local administrator account..."
Write-Host "Choose your own password (at least 12 characters). It will not be stored in Git."
& $Python -m student_performance.manage create-user --email admin@mtu.local --name "MTU Administrator" --role admin

Write-Host "[6/6] Adding synthetic demonstration records..."
& $Python scripts/seed_demo_system.py --rows 18

Write-Host ""
Write-Host "Setup complete. Start the system with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\start_local_web.ps1"
Write-Host "Then open http://localhost:3000 and sign in as admin@mtu.local."
