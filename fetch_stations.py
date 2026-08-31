import os
import sys
import requests
import pandas as pd


def get_supabase_config():
    base_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not base_url:
        print("ERROR: SUPABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)

    if not service_key:
        print("ERROR: SUPABASE_SERVICE_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    return base_url.rstrip("/"), service_key


def fetch_station_list():
    url = (
        "https://data.environment.alberta.ca/"
        "EdwServices/aqhi/odata/Stations"
        "?$select=Name,Latitude,Longitude"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw = response.json()

    df = pd.json_normalize(raw["value"])[
        ["Name", "Latitude", "Longitude"]
    ]

    df = df.dropna(
        subset=["Name", "Latitude", "Longitude"]
    )

    df["Name"] = df["Name"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["Name"])

    df = df.rename(
        columns={"Name": "StationName"}
    )

    return df[
        ["StationName", "Latitude", "Longitude"]
    ]


def upsert_stations(df):
    base_url, service_key = get_supabase_config()

    url = f"{base_url}/rest/v1/stations"

    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    records = df.to_dict(orient="records")

    response = requests.post(
        url,
        headers=headers,
        params={"on_conflict": "StationName"},
        json=records,
        timeout=60,
    )

    if response.status_code not in (200, 201, 204):
        print(
            f"ERROR: Supabase upsert failed: "
            f"{response.status_code} {response.text}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    try:
        stations = fetch_station_list()

        if stations.empty:
            print("No stations returned.")
            sys.exit(1)

        upsert_stations(stations)

        print(
            f"Successfully upserted "
            f"{len(stations)} stations."
        )

    except requests.RequestException as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
