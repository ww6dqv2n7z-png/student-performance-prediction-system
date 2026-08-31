#!/bin/zsh
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="$project_dir/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found. Follow the README setup steps first."
  exit 1
fi

if [[ ! -d "$project_dir/web/node_modules" ]]; then
  echo "Website dependencies not found. Run: cd web && npm install"
  exit 1
fi

cd "$project_dir"
"$python_bin" -m student_performance.api &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM

cd "$project_dir/web"
npm run dev
