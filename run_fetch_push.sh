#!/bin/bash
set -e

source /opt/airquality/venv/bin/activate

set -a
source /opt/airquality/config/intelligence.env
set +a

cd /opt/airquality/github/AB_datapull

LOCKFILE="/opt/airquality/locks/ab_datapull_git.lock"
mkdir -p "$(dirname "$LOCKFILE")"

(
  flock -w 600 200
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

  python fetch_push.py

  git add .

  if git diff --cached --quiet; then
      echo "No changes to commit."
  else
      git commit -m "Update fetch_push $(date '+%Y-%m-%d %H:%M')"
      for attempt in 1 2 3; do
          if git push origin main; then
              break
          fi
          echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
          git pull --rebase origin main
      done
  fi
) 200>"$LOCKFILE"
