#!/bin/bash

# =================================================================
# FedMI Reproducibility Script
# =================================================================
# This script runs the full automated experiment suite defined in
# experiments/automated_suite.py.
#
# It includes:
# 1. An IID Baseline experiment.
# 2. A Non-IID (Label Skew) experiment.
#
# All results will be archived in the 'exps/' directory.
# =================================================================

echo "Starting FedMI Reproducibility Suite..."
echo "Date: $(date)"
echo "----------------------------------------------------------------"

# Ensure we are in the root directory
# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming script is in experiments/, go up one level to root
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"
echo "Working Directory: $(pwd)"

# Run the automated suite python script
python3 experiments/automated_suite.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "Suite completed successfully."
else
    echo "----------------------------------------------------------------"
    echo "Suite failed with exit code $exit_code."
fi

exit $exit_code
