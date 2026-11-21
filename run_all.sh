#!/usr/bin/env bash
# Runs the full pipeline in order. Usage (from the repository root): bash run_all.sh
set -e
PY=python
command -v python >/dev/null 2>&1 || PY=python3   # macOS often has only python3
$PY SCRIPTS/01_check_data_and_build_features.py
$PY SCRIPTS/02_exploratory_analysis.py
$PY SCRIPTS/03_repeated_cross_validation.py
$PY SCRIPTS/04_leave_one_site_out.py
$PY SCRIPTS/05_motion_sensitivity.py
$PY SCRIPTS/06_results_summary.py
