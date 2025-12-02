# src/features/target_transforms.py

import numpy as np
import pandas as pd


def log_transform_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Apply log1p transform to the target column in-place:
        y_log = log(1 + y)

    This is useful for long-tailed targets like trip_duration.
    """
    if target_col not in df.columns:
        return df

    df[target_col] = np.log1p(df[target_col].astype(float))
    return df


def inverse_log_transform(y_log):
    """
    Inverse of log1p: y = exp(y_log) - 1

    You can use this in evaluation or prediction if you want metrics 
    back in the original scale.
    """
    return np.expm1(y_log)
