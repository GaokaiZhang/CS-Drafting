#!/bin/bash
set -euo pipefail

RUN_ROOT="/home/$USER/repo/mlsys/CS-Drafting"

eval "$(conda shell.bash hook)"
conda activate mlsys-fw

cd "$RUN_ROOT"
python flask_ui.py
