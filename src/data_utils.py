# data_utils.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import yaml

from .features.datetime_features import add_datetime_features
from .features.spatial_features import add_haversine_distance
from .features.target_transforms import log_transform_target


CONFIG_DATASETS_DIR = Path("configs/datasets")


def load_dataset_config(name: str) -> Dict[str, Any]:
    """
    Load dataset configuration from configs/datasets/<name>.yml
    """
    path = CONFIG_DATASETS_DIR / f"{name}.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_preprocessor(
    df: pd.DataFrame,
    keep_categoricals: bool = False,
    scale_numeric: bool = True,
):
    """
    Build a ColumnTransformer that:
      - imputes numeric columns (median)
      - optionally scales numeric columns
      - optionally imputes + one-hot encodes categorical columns

    Note: current semantics:
      - if keep_categoricals is True → OHE categoricals
      - if keep_categoricals is False → only numeric columns are used
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist() if keep_categoricals else []

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    numeric_pipe = Pipeline(numeric_steps)

    if keep_categoricals and cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
    else:
        pre = ColumnTransformer(
            transformers=[("num", numeric_pipe, num_cols)],
            remainder="drop",
            verbose_feature_names_out=False
        )
    return pre

def load_data(
    path: str,
    target: str,
    test_size: float = 0.25,
    random_state: int = 42,
    *,
    drop_cols: Optional[list[str]] = None,
    keep_categoricals: bool = False,
    scale_numeric: bool = True,
    time_split: bool = False,
    datetime_col: Optional[str] = None,
    feature_engineering: Optional[Dict[str, Any]] = None,
    spatial_cols: Optional[Dict[str, str]] = None,
    rename_cols: Optional[Dict[str, str]] = None,
    return_preprocessor: bool = True,
    return_raw: bool = True
    ):
    """
    Load and preprocess data from a CSV path according to the given options.

    Backward-compatible:
    - Old: X_train, X_test, y_train, y_test
    - New (if return_preprocessor=True): also returns (preprocessor, feature_names)
    - New (if return_raw=True): also returns (X_train_df, X_test_df)

    New capabilities via `feature_engineering`:
      feature_engineering = {
        "datetime": True/False,
        "haversine": True/False,
        "log_target": True/False,
      }
    """
    fe = feature_engineering or {}

    df = pd.read_csv(path)

    # 0) Column renaming (dataset-specific)
    if rename_cols:
        df = df.rename(columns=rename_cols)
    print("************************************************************")
    print("rename columns:", rename_cols)
    print("df columns after rename:", list(df.columns))
    print("************************************************************")
    
    # From here on, we only use the CLEAN names
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")

    
    # 1) Target transform (e.g., log1p) BEFORE splitting
    if fe.get("log_target"):
        df = log_transform_target(df, target)

    # 2) Separate target
    y = df.pop(target)

    # 3) Datetime-based feature engineering (does not drop datetime_col)
    if fe.get("datetime") and datetime_col:
        if datetime_col not in df.columns:
            raise ValueError(
                f"datetime_col '{datetime_col}' specified, but not found in data."
            )
        df = add_datetime_features(df, datetime_col)

    # 4) Spatial / distance features
    if fe.get("haversine"):
        df = add_haversine_distance(df, spatial_cols=spatial_cols)

    # 5) Train/test split
    if time_split:
        if not datetime_col or datetime_col not in df.columns:
            raise ValueError(
                "time_split=True requires a valid datetime_col present in the CSV."
                )
        # Sort by datetime and split last fraction as test
        dt_series = pd.to_datetime(df[datetime_col])
        order = np.argsort(dt_series.values)

        df_sorted = df.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)

        n = len(df_sorted)
        cut = int(n * (1 - test_size))
        X_train_df, X_test_df = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
        y_train, y_test = y_sorted.iloc[:cut], y_sorted.iloc[cut:]
    else:
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=random_state
        )
    # 6) Apply dataset-specific drop columns
    if drop_cols:
        cols_to_drop = [c for c in drop_cols if c in X_train_df.columns]
        X_train_df = X_train_df.drop(columns=cols_to_drop, errors="ignore")
        X_test_df = X_test_df.drop(columns=cols_to_drop, errors="ignore")
        print("************************************************************")
        print("df columns after drope:", list(X_test_df.columns))
        print("************************************************************")

    # 7) Build & apply preprocessing (handles NaNs; scales if requested; optional OHE)
    pre = _build_preprocessor(
        X_train_df, 
        keep_categoricals=keep_categoricals, 
        scale_numeric=scale_numeric)
    X_train = pre.fit_transform(X_train_df)
    X_test = pre.transform(X_test_df)

    # 8) Feature names (best-effort)
    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    # 9) Build outputs
    outputs = [X_train, X_test, y_train, y_test]
    if return_preprocessor:
        outputs.extend([pre, feature_names])
    if return_raw:
        outputs.extend([X_train_df, X_test_df])

    return tuple(outputs)