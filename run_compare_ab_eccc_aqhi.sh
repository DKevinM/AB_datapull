#!/bin/bash
set -e

source /opt/airquality/venv/bin/activate
cd /opt/airquality/github/AB_datapull

git fetch origin
git pull --rebase origin main

python scripts/compare_ab_eccc_aqhi.py

git add data/aqhi_comparison/*.csv

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "Update AB vs ECCC AQHI comparison"
git push origin main
