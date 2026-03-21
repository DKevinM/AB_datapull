#!/usr/bin/env python3
"""
Health check: verify data freshness and pipeline health.

Exits with code 0 if healthy, 1 if stale data detected.

Usage:
    python scripts/health_check.py [--max-age-mins 45]
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_MAX_AGE_MINS = 45


def main():
    parser = argparse.ArgumentParser(description="Check data freshness in Supabase.")
    parser.add_argument(
        "--max-age-mins",
        type=int,
        default=DEFAULT_MAX_AGE_MINS,
        help=f"Alert if data older than this many minutes (default: {DEFAULT_MAX_AGE_MINS})",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Running health check...")

    try:
        from src.api.queries import SupabaseQueries
        queries = SupabaseQueries()
    except Exception as e:
        logger.error(f"Could not connect to Supabase: {e}")
        sys.exit(1)

    # Check latest data timestamp
    latest_ts = queries.get_latest_reading_timestamp()
    if latest_ts is None:
        logger.error("HEALTH CHECK FAILED: No data found in Supabase sensor_readings table")
        sys.exit(1)

    now_utc = datetime.now(timezone.utc)
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)

    age_mins = (now_utc - latest_ts).total_seconds() / 60
    logger.info(f"Latest reading: {latest_ts.isoformat()} ({age_mins:.1f} minutes ago)")

    # Check active sensor count
    active_count = queries.get_active_sensor_count(hours=1)
    logger.info(f"Active sensors (last 1h): {active_count}")

    # Evaluate health
    if age_mins > args.max_age_mins:
        logger.error(
            f"HEALTH CHECK FAILED: Data is stale — {age_mins:.1f} mins old "
            f"(threshold: {args.max_age_mins} mins)"
        )
        sys.exit(1)

    if active_count == 0:
        logger.warning("HEALTH CHECK WARNING: No active sensors in the last hour")

    logger.info(f"HEALTH CHECK PASSED: Data is fresh ({age_mins:.1f} mins old), {active_count} active sensors")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
