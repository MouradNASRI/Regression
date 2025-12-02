# src/features/datetime_features.py

import pandas as pd


def add_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Add basic calendar/time features based on a datetime column.

    Features added:
      - <col>_hour   : integer hour of day [0–23]
      - <col>_dow    : day of week [0=Monday, 6=Sunday]
      - <col>_month  : month [1–12]
      - <col>_is_weekend: 1 if Saturday/Sunday, else 0

    The original datetime column is kept as-is (your drop_cols can remove it
    later if you wish).
    """
    if datetime_col not in df.columns:
        return df

    dt = pd.to_datetime(df[datetime_col])

    base = datetime_col
    df[f"{base}_hour"] = dt.dt.hour
    df[f"{base}_dow"] = dt.dt.weekday
    df[f"{base}_month"] = dt.dt.month
    df[f"{base}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)

    return df
