# src/features/spatial_features.py

import numpy as np
import pandas as pd
from typing import Optional, Dict


def haversine_distance(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Compute the great-circle distance between two points on Earth (in km)
    using the Haversine formula.

    All inputs are in degrees.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_haversine_distance(
    df: pd.DataFrame,
    spatial_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Add a 'distance_km' feature based on pickup/dropoff lat/lon.

    Parameters
    ----------
    df : DataFrame
        Input data.
    spatial_cols : dict or None
        Mapping of logical spatial roles to column names:
            {
                "pickup_lat": "<col_name>",
                "pickup_lon": "<col_name>",
                "dropoff_lat": "<col_name>",
                "dropoff_lon": "<col_name>",
            }

        If None, defaults to NYC taxi naming:
            pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude

    Behavior
    --------
    - If any required column is missing, the function returns df unchanged.
    """
    if spatial_cols is None:
        spatial_cols = {
            "pickup_lat": "pickup_latitude",
            "pickup_lon": "pickup_longitude",
            "dropoff_lat": "dropoff_latitude",
            "dropoff_lon": "dropoff_longitude",
        }

    required = [
        spatial_cols.get("pickup_lat"),
        spatial_cols.get("pickup_lon"),
        spatial_cols.get("dropoff_lat"),
        spatial_cols.get("dropoff_lon"),
    ]

    if any(c is None for c in required):
        # Misconfigured spatial_cols; nothing to do
        return df

    if not all(c in df.columns for c in required):
        # Columns not present in this dataset; do nothing
        return df

    lat1 = df[spatial_cols["pickup_lat"]]
    lon1 = df[spatial_cols["pickup_lon"]]
    lat2 = df[spatial_cols["dropoff_lat"]]
    lon2 = df[spatial_cols["dropoff_lon"]]

    df = df.copy()
    df["distance_km"] = haversine_distance(lat1, lon1, lat2, lon2)
    return df
