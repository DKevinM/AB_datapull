name: Sync Stations Daily
on:
  schedule:
    - cron: "5 9 * * *"   # 09:05 UTC daily
  workflow_dispatch: {}

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install requests pandas SQLAlchemy psycopg2-binary
      - name: Update stations table
        run: python path/to/sync_stations.py
        env:
          SUPABASE_DB_URL: ${{ secrets.SUPABASE_DB_URL }}
