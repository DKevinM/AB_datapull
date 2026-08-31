#!/bin/bash

set -e

source /opt/airquality/venv/bin/activate

cd /opt/airquality/github/AB_datapull

echo "========================================"
echo "AQHI IDW Run: $(date)"
echo "========================================"

git fetch origin
git pull --rebase origin main

python AQHI_idw.py

git add data/

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "Update AQHI IDW grid (CSV + GeoJSON)"

git push origin main

echo "Finished."
