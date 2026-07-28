"""
Feature engineering for measles outbreak prediction.

Creates:
- Lag features (1, 2, 4, 8 weeks)
- Rolling averages (4-week, 12-week)
- Rolling standard deviation (volatility)
- Season (week of year)
- Year trend
- Vaccination rate
- Target: cases_next_4w (regression), outbreak_risk (classification)
"""
import pandas as pd
import numpy as np


OUTBREAK_THRESHOLDS = {"low": 10, "medium": 50}  # cases per 4 weeks


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # --- lag features ---
    for lag in [1, 2, 4, 8]:
        df[f"cases_lag_{lag}w"] = df["cases"].shift(lag)

    # --- rolling features ---
    df["rolling_mean_4w"] = df["cases"].shift(1).rolling(4).mean()
    df["rolling_mean_12w"] = df["cases"].shift(1).rolling(12).mean()
    df["rolling_std_4w"] = df["cases"].shift(1).rolling(4).std()
    df["rolling_max_4w"] = df["cases"].shift(1).rolling(4).max()

    # trend: ratio of recent avg vs longer avg (>1 = rising)
    df["trend_ratio"] = (df["rolling_mean_4w"] / (df["rolling_mean_12w"] + 0.1)).clip(0, 10)

    # --- time features ---
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["year_trend"] = df["year"] - df["year"].min()

    # seasonal: sin/cos encoding of week (captures cyclical nature)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # --- targets ---
    # regression: total cases in next 4 weeks
    df["cases_next_4w"] = df["cases"].shift(-4).rolling(4).sum().shift(-3) # noqa

    # simpler: just next week's cases
    df["cases_next_1w"] = df["cases"].shift(-1)

    # 4-week forward sum (what we actually predict)
    future_4w = df["cases"].shift(-1) + df["cases"].shift(-2) + \
                df["cases"].shift(-3) + df["cases"].shift(-4)
    df["cases_next_4w"] = future_4w

    # classification: outbreak risk based on next 4 weeks
    def risk_label(val):
        if pd.isna(val):
            return np.nan
        if val <= OUTBREAK_THRESHOLDS["low"]:
            return 0  # low
        elif val <= OUTBREAK_THRESHOLDS["medium"]:
            return 1  # medium
        else:
            return 2  # high

    df["outbreak_risk"] = df["cases_next_4w"].apply(risk_label)

    return df


FEATURE_COLS = [
    "cases_lag_1w", "cases_lag_2w", "cases_lag_4w", "cases_lag_8w",
    "rolling_mean_4w", "rolling_mean_12w", "rolling_std_4w", "rolling_max_4w",
    "trend_ratio", "week_sin", "week_cos", "year_trend", "vax_rate",
]

TARGET_REG = "cases_next_4w"
TARGET_CLF = "outbreak_risk"
