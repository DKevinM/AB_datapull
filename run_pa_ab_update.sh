#!/bin/bash
set -e


source /opt/airquality/venv/bin/activate


set -a
source /opt/airquality/config/intelligence.env
set +a


SELF_LOCK="/opt/airquality/locks/run_pa_ab_update.selflock"
mkdir -p "$(dirname "$SELF_LOCK")"
exec 201>"$SELF_LOCK"
if ! flock -n 201; then
    echo "Previous run_pa_ab_update.sh instance still running; skipping this cycle."
    exit 0
fi

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

  python AB_PA_latest.py
  python scripts/build_eAQHI.py
  # BC 150 km border band (PA_BC_pull.py) - map/Supabase only, kept out of AB products; never blocks AB
  python AB_PA_latest.py BC || echo "WARN: BC PurpleAir pull failed"

  git add data/AB_PM25_map.json
  git add data/eAQHI_map.json
  git add data/BC_PM25_map.json 2>/dev/null || true

  if git diff --cached --quiet; then
      echo "No changes to commit."
  else
      git commit -m "Update PurpleAir + eAQHI $(date +'%Y-%m-%d %H:%M')"
      for attempt in 1 2 3; do
          if git push origin main; then
              break
          fi
          echo "push rejected (attempt $attempt/3); rebasing onto latest and retrying..."
          git pull --rebase origin main
      done
  fi
) 200>"$LOCKFILE"
