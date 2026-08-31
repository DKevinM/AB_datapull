#!/bin/bash
set -e

source /opt/airquality/venv/bin/activate
cd /opt/airquality/github/AB_datapull

LOCKFILE="/opt/airquality/locks/ab_datapull_git.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -w 600 200

  git fetch origin
  git pull --rebase origin main

  python fetch_last6h.py

  git add data/last6h.csv
  git add data/stations.geojson

  if git diff --cached --quiet; then
      echo "No changes to commit."
      exit 0
  fi

  git commit -m "chore: update AQHI data (csv + geojson)"
  for attempt in 1 2 3; do
      if git push origin main; then
          break
      fi
      echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
      git pull --rebase origin main
  done
) 200>"$LOCKFILE"
