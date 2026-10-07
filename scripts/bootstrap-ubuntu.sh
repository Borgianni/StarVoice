#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3 python3-venv git libopus0 libopus-dev

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,analysis]"

starvoice doctor
