#!/bin/bash

set -e

source /opt/airquality/venv/bin/activate

cd /opt/airquality/github/AB_datapull

echo "========================================"
echo "ACA AQHI IDW Run: $(date)"
echo "========================================"

LOCKFILE="/opt/airquality/locks/ab_datapull_git.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -w 120 200
  for attempt in 1 2 3 4 5; do
      if git fetch origin && git pull --rebase origin main; then
          break
      fi
      if [ "$attempt" -eq 5 ]; then
          echo "ERROR: git pull failed after 5 attempts."
          exit 1
      fi
      echo "pull failed (attempt $attempt/5); retrying in 15s..."
      sleep 15
  done
) 200>"$LOCKFILE"

python AQHI_ACA_idw.py

(
  flock -w 120 200

  git add data/

  if git diff --cached --quiet; then
      echo "No changes to commit."
  else
      git commit -m "Update ACA AQHI IDW grid"
      for attempt in 1 2 3; do
          if git push origin main; then
              break
          fi
          echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
          git pull --rebase origin main
      done
  fi
) 200>"$LOCKFILE"

echo "Finished."
