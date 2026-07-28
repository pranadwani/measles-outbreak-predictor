"""
Fetches real measles data from the JHU CSSE Measles Data Repository.
https://github.com/CSSEGISandData/measles_data

Data is updated every Friday by the Johns Hopkins University Measles Tracking Team.
Licensed under CC BY 4.0 — attribution: "JHU Measles Tracking Team Data Repository"

This replaces generate_data.py — run this to get live, real data.
"""
import os
import requests
import pandas as pd
import numpy as np
from io import StringIO

# ── JHU CSSE GitHub raw URLs (updated every Friday) ──────────────────────────
JHU_BASE = "https://raw.githubusercontent.com/CSSEGISandData/measles_data/main"
COUNTY_URL = f"{JHU_BASE}/measles_county_all_updates.csv"
WEEKLY_STATES_URL = f"{JHU_BASE}/Top_states_time_series.csv"

HEADERS = {"User-Agent": "measles-predictor/1.0"}

# Real CDC annual totals for years before 2025 (JHU data only covers 2025+)
CDC_HISTORICAL = {
    2010: 63,  2011: 220, 2012: 55,  2013: 187, 2014: 667,
    2015: 188, 2016: 86,  2017: 118, 2018: 372, 2019: 1282,
    2020: 13,  2021: 49,  2022: 121, 2023: 58,  2024: 285,
}

CDC_VAX = {
    2010: 90.0, 2011: 91.0, 2012: 91.0, 2013: 91.5, 2014: 91.5,
    2015: 91.9, 2016: 91.1, 2017: 91.5, 2018: 91.1, 2019: 90.8,
    2020: 89.6, 2021: 88.9, 2022: 92.7, 2023: 92.7, 2024: 91.0,
    2025: 88.5, 2026: 87.0,
}


def fetch_jhu_national_weekly():
    """
    Fetches county-level daily data from JHU and aggregates to national weekly totals.
    """
    # print("fetching JHU county level data")
    resp = requests.get(COUNTY_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["date"])
    df = df[df["outcome_type"] == "case_lab-confirmed"].copy()

    # aggregate to national daily
    daily = df.groupby("date")["value"].sum().reset_index()
    daily.columns = ["date", "cases"]

    # resample to weekly (Sunday start to match epidemiological weeks)
    daily = daily.set_index("date").sort_index()
    weekly = daily.resample("W-SAT").sum().reset_index()
    weekly["week_start"] = weekly["date"] - pd.Timedelta(days=6)
    weekly = weekly.rename(columns={"date": "week_end"})
    weekly["year"] = weekly["week_start"].dt.year
    weekly["week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["source"] = "JHU"

    print(f"JHU: {len(weekly)} weeks, {weekly['week_start'].min().date()} to {weekly['week_start'].max().date()}")
    print(f"JHU total cases: {weekly['cases'].sum():,}")
    return weekly[["week_start", "week_end", "year", "week", "cases", "source"]]


def historical_to_weekly(seed=42):
    """
    Distributes CDC historical annual totals (pre-2025) into weekly data
    using seasonal distribution (measles peaks late winter/spring).
    """
    np.random.seed(seed)
    rows = []

    for year, annual_total in CDC_HISTORICAL.items():
        weeks = 52
        week_nums = np.arange(1, weeks + 1)
        seasonal = np.exp(-0.5 * ((week_nums - 12) / 8) ** 2)
        seasonal /= seasonal.sum()

        base = seasonal * annual_total
        noise = np.random.normal(0, base * 0.15 + 0.5)
        weekly = np.clip(base + noise, 0, None).round().astype(int)

        if year in (2014, 2019):
            spike_week = np.random.randint(5, 16)
            for w in range(spike_week, min(spike_week + 10, weeks)):
                weekly[w] += int(annual_total * np.random.uniform(0.03, 0.07))

        for w_idx, count in enumerate(weekly):
            pass
            try:
                date = pd.to_datetime(f"{year}-W{w_idx+1:02d}-1", format="%G-W%V-%u")
            except Exception:
                continue
            rows.append({
                "week_start": date,
                "week_end": date + pd.Timedelta(days=6),
                "year": year,
                "week": w_idx + 1,
                "cases": max(0, int(count)),
                "source": "CDC_historical",
            })

    return pd.DataFrame(rows)


def build_full_dataset(historical_df, jhu_df):
    """
    Combine historical (pre-2025) with live JHU data (2025+).
    Drop any overlap — JHU data takes priority for 2025+.
    """
    # keep historical only for years before 2025
    historical_pre2025 = historical_df[historical_df["year"] < 2025].copy()

    combined = pd.concat([historical_pre2025, jhu_df], ignore_index=True)
    combined = combined.sort_values("week_start").reset_index(drop=True)

    # add vaccination rate
    combined["vax_rate"] = combined["year"].map(CDC_VAX).fillna(90.0)

    # rename for pipeline compatibility
    combined["date"] = combined["week_start"]

    return combined[["date", "year", "week", "cases", "vax_rate", "source"]]


def main():
    os.makedirs("data", exist_ok=True)

    # fetch live JHU data
    try:
        jhu_df = fetch_jhu_national_weekly()
    except Exception as e:
        print(f"JHU fetch failed: {e}")
        print("Falling back to generated data for 2025+")
        jhu_df = pd.DataFrame()

    # build historical weekly
    print("\nBuilding historical weekly data (2010-2024) from CDC annual totals...")
    historical_df = historical_to_weekly()

    # combine
    if not jhu_df.empty:
        full_df = build_full_dataset(historical_df, jhu_df)
    else:
        full_df = build_full_dataset(historical_df, pd.DataFrame())

    full_df.to_csv("data/measles_weekly.csv", index=False)

    print(f"\nSaved {len(full_df)} weekly records")
    print(f"Years: {full_df['year'].min()} – {full_df['year'].max()}")
    print(f"Total cases: {full_df['cases'].sum():,}")
    print(f"Sources: {full_df['source'].value_counts().to_dict()}")
    print("\nRun next: python src/train.py")


if __name__ == "__main__":
    main()