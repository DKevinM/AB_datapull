import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, UTC
from pathlib import Path
from io import StringIO

OUTDIR = Path("data/aqhi_comparison")
OUTDIR.mkdir(parents=True, exist_ok=True)

AB_COMMUNITY_URL = (
    "https://data.environment.alberta.ca/EdwServices/aqhi/odata/"
    "CommunityAqhis?$format=json"
)

ECCC_AB_SUMMARY_URL = "https://weather.gc.ca/airquality/pages/provincial_summary/ab_e.html"

COMMUNITY_MATCH = {
    "Edmonton": "Edmonton",
    "Calgary": "Calgary",
    "Red Deer": "Red Deer",
    "Fort McMurray": "Fort McMurray",
    "Grande Prairie": "Grande Prairie",
    "Lethbridge": "Lethbridge",
    "Medicine Hat": "Medicine Hat",
    "Airdrie": "Airdrie",
    "Cold Lake": "Cold Lake",
    "Drayton Valley": "Drayton Valley",
    "St. Albert": "St. Albert",
    "Strathcona County": "Sherwood Park",
}


def parse_aqhi_value(x):
    if pd.isna(x):
        return None

    txt = str(x).strip()

    if txt.upper() in ["N/A", "NA", ""]:
        return None

    if "10+" in txt:
        return 11

    m = re.search(r"\d+", txt)
    return int(m.group()) if m else None


def pull_ab_community_aqhi():
    r = requests.get(AB_COMMUNITY_URL, timeout=30)
    r.raise_for_status()

    data = r.json()["value"]
    df = pd.DataFrame(data)

    # Adjust these if Alberta changes field names
    keep_cols = [
        c for c in df.columns
        if c.lower() in [
            "communityname",
            "community",
            "aqhi",
            "readingdate",
            "forecasttoday",
            "forecasttonight",
            "forecasttomorrow"
        ]
    ]

    df = df[keep_cols].copy()

    if "CommunityName" in df.columns:
        df = df.rename(columns={"CommunityName": "community_ab"})
    elif "Community" in df.columns:
        df = df.rename(columns={"Community": "community_ab"})

    df = df.rename(columns={
        "Aqhi": "ab_current_aqhi",
        "AQHI": "ab_current_aqhi",
        "ReadingDate": "ab_reading_date",
        "ForecastToday": "ab_forecast_today",
        "ForecastTonight": "ab_forecast_tonight",
        "ForecastTomorrow": "ab_forecast_tomorrow",
    })

    df["ab_current_aqhi"] = df["ab_current_aqhi"].apply(parse_aqhi_value)
    df["ab_pulled_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")

    return df


def pull_eccc_ab_summary():

    r = requests.get(
        ECCC_AB_SUMMARY_URL,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    print(f"\nFOUND {len(tables)} TABLES FROM ECCC\n")

    aqhi_table = None

    # =====================================================
    # FIND TABLE CONTAINING AQHI DATA
    # =====================================================

    for i, tbl in enumerate(tables):

        test = tbl.copy()

        # Flatten columns
        test.columns = [
            " ".join([str(x) for x in col if str(x) != "nan"]).strip()
            if isinstance(col, tuple)
            else str(col)
            for col in test.columns
        ]

        cols_lower = [c.lower() for c in test.columns]

        print(f"\nTABLE {i}")
        print(test.head())
        print(test.columns.tolist())

        # Look for AQHI-style columns
        if any(
            ("observed" in c)
            or ("current" in c)
            or ("aqhi" in c)
            for c in cols_lower
        ):
            aqhi_table = test
            print(f"\nUSING TABLE {i} AS AQHI TABLE\n")
            break

    if aqhi_table is None:
        print("\nNO AQHI TABLE FOUND\n")
        return pd.DataFrame(columns=[
            "community_eccc",
            "eccc_current_aqhi",
            "eccc_pulled_at_utc"
        ])

    # =====================================================
    # IDENTIFY COLUMNS
    # =====================================================

    location_col = aqhi_table.columns[0]

    observed_col = None

    for col in aqhi_table.columns:

        c = col.lower()

        if (
            "observed" in c
            or "current" in c
        ):
            observed_col = col
            break

    # fallback
    if observed_col is None:
        observed_col = aqhi_table.columns[1]

    print(f"\nLOCATION COLUMN: {location_col}")
    print(f"OBSERVED COLUMN: {observed_col}\n")

    eccc = aqhi_table.rename(columns={
        location_col: "community_eccc",
        observed_col: "eccc_current_aqhi"
    })

    eccc = eccc[[
        "community_eccc",
        "eccc_current_aqhi"
    ]].copy()

    # =====================================================
    # CLEAN COMMUNITY NAMES
    # =====================================================

    eccc["community_eccc"] = (
        eccc["community_eccc"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*Alberta\s*", "", regex=True)
        .str.strip()
    )

    # =====================================================
    # CLEAN AQHI VALUES
    # =====================================================

    eccc["eccc_current_aqhi"] = (
        eccc["eccc_current_aqhi"]
        .apply(parse_aqhi_value)
    )

    # =====================================================
    # FILTER TO MATCHED COMMUNITIES
    # =====================================================

    eccc = eccc[
        eccc["community_eccc"].isin(COMMUNITY_MATCH.values())
    ].copy()

    eccc["eccc_pulled_at_utc"] = (
        datetime.now(UTC).isoformat(timespec="seconds")
    )

    print("\nFINAL ECCC DATASET\n")
    print(eccc)

    return eccc



def compare_aqhi():
    ab = pull_ab_community_aqhi()
    eccc = pull_eccc_ab_summary()

    match = pd.DataFrame([
        {"community_ab": ab_name, "community_eccc": eccc_name}
        for ab_name, eccc_name in COMMUNITY_MATCH.items()
    ])

    combined = (
        match
        .merge(ab, on="community_ab", how="left")
        .merge(eccc, on="community_eccc", how="left", suffixes=("", "_eccc"))
    )

    combined["diff_ab_minus_eccc"] = (
        combined["ab_current_aqhi"] - combined["eccc_current_aqhi"]
    )

    combined["abs_diff"] = combined["diff_ab_minus_eccc"].abs()

    combined["flag"] = "OK"
    combined.loc[combined["abs_diff"] >= 1, "flag"] = "Minor difference"
    combined.loc[combined["abs_diff"] >= 2, "flag"] = "Significant difference"
    combined.loc[combined["abs_diff"] >= 3, "flag"] = "Major difference"

    return combined


if __name__ == "__main__":
    df = compare_aqhi()

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    month = datetime.now(UTC).strftime("%Y-%m")

    daily_file = OUTDIR / f"aqhi_compare_{today}.csv"
    monthly_file = OUTDIR / f"aqhi_compare_{month}.csv"

    df.to_csv(daily_file, index=False)

    if monthly_file.exists():
        old = pd.read_csv(monthly_file)
        df_month = pd.concat([old, df], ignore_index=True)
        df_month = df_month.drop_duplicates(
            subset=["ab_pulled_at_utc", "community_ab"],
            keep="last"
        )
    else:
        df_month = df

    df_month.to_csv(monthly_file, index=False)

    summary = (
        df_month
        .groupby("community_ab", dropna=False)
        .agg(
            n_records=("community_ab", "count"),
            mean_diff=("diff_ab_minus_eccc", "mean"),
            max_abs_diff=("abs_diff", "max"),
            hours_with_difference=("abs_diff", lambda x: (x >= 1).sum()),
            significant_differences=("abs_diff", lambda x: (x >= 2).sum()),
        )
        .reset_index()
    )

    summary_file = OUTDIR / f"aqhi_compare_summary_{month}.csv"
    summary.to_csv(summary_file, index=False)

    print("\nCurrent comparison:")
    print(df[[
        "community_ab",
        "community_eccc",
        "ab_current_aqhi",
        "eccc_current_aqhi",
        "diff_ab_minus_eccc",
        "flag"
    ]].to_string(index=False))

    print(f"\nSaved:")
    print(daily_file)
    print(monthly_file)
    print(summary_file)
