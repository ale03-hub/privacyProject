#!/usr/bin/env bash
# First-time setup: virtualenv + dependencies
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
