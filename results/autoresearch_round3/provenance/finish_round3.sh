#!/usr/bin/env bash
set -euo pipefail
cd /home/enmingzz/project/vla/frequency_vla
source scripts/env.sh
export MPLBACKEND=Agg
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" -m frequency_vla.study_analysis --results-dir results/autoresearch_round3
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH "$LIBERO_VENV/bin/python" .runtime/finish_round3.py
