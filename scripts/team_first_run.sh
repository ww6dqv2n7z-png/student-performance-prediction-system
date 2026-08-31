#!/bin/zsh
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.10, 3.11, or 3.12 first."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required. Install the current Node.js LTS release first."
  exit 1
fi

python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12"'

echo "[1/6] Creating the Python environment..."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .

echo "[2/6] Installing website dependencies..."
npm --prefix web install

echo "[3/6] Generating the synthetic CEIT demonstration dataset..."
.venv/bin/python scripts/generate_ceit_dataset.py \
  --records-per-level 200 \
  --seed 20260831 \
  --output data/ceit_synthetic_students.csv

echo "[4/6] Training the ANN demonstration model..."
.venv/bin/python -m student_performance.train \
  --data data/ceit_synthetic_students.csv \
  --dataset-label "Synthetic MTU CEIT Project Dataset" \
  --synthetic-data \
  --task classification \
  --pass-threshold 50 \
  --output artifacts/classifier

echo "[5/6] Creating your local administrator account..."
echo "Choose your own password (at least 12 characters). It will not be stored in Git."
.venv/bin/python -m student_performance.manage create-user \
  --email admin@mtu.local \
  --name "MTU Administrator" \
  --role admin

echo "[6/6] Adding synthetic demonstration records..."
.venv/bin/python scripts/seed_demo_system.py --rows 18

echo ""
echo "Setup complete. Start the system with:"
echo "  ./scripts/start_local_web.sh"
echo "Then open http://localhost:3000 and sign in as admin@mtu.local."
