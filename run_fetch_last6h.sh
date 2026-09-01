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
          # LiveMap reads last6h.csv via jsdelivr's GitHub mirror, not
          # raw.githubusercontent.com - its edge nodes lag origin
          # independently with no way for us to force a refresh, which was
          # actively misleading users on displayed data age. jsdelivr caches
          # @main aggressively too, but exposes a real purge API - call it
          # every push so staleness here stays bounded to this request
          # instead of an unpredictable multi-hour edge lag.
          curl -s -o /dev/null "https://purge.jsdelivr.net/gh/DKevinM/AB_datapull@main/data/last6h.csv" || true
          break
      fi
      echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
      git pull --rebase origin main
  done
) 200>"$LOCKFILE"
