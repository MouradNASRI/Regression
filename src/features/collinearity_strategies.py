"""
===============================================================================
  Collinearity Reduction Strategies — Strategy Pattern
===============================================================================

This module implements a set of reusable, model-agnostic collinearity
detection and reduction strategies (Pearson, VIF, SVD, condition number).

Design goals
------------
- Keep each strategy small and focused (single responsibility).
- Use a common interface so strategies can be chained by a manager.
- Stay dataset-agnostic: only assume a numeric pandas.DataFrame X.
- Be easy to log to MLflow via a structured diagnostics object.
- Avoid hidden side-effects and "magic" global state.

Intended usage
--------------
- The training pipeline calls a CollinearityManager (in another module).
- The manager instantiates one or more reducers from this file.
- Each reducer:
    X_reduced, diagnostics = reducer.fit_transform(X, y)

Where:
- X_reduced is the possibly-modified design matrix.
- diagnostics describes what happened (dropped features, notes, extra stats).

This file does NOT know which model (OLS, Ridge, etc.) is used.
That decision happens at the model-config / pipeline level.
===============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =============================================================================
# Diagnostics container
# =============================================================================

@dataclass
class CollinearityDiagnostics:
    """
    Structured result returned by each collinearity reducer.

    The goal is to provide enough information for:
      - MLflow logging
      - human-readable reporting
      - post-hoc analysis and debugging

    Attributes
    ----------
    dropped_features : List[str]
        Names of features removed by this strategy. This list may be empty if
        the reducer is "diagnostic-only" (e.g., SVD or condition number).

    notes : List[str]
        Short, human-readable messages describing the decisions taken.
        Example: "Pearson | feature_a–feature_b corr=0.98 → dropped feature_b"

    extra : Dict[str, Any]
        Any additional structured data that may be useful for logging or
        inspection. Suggested keys:
          - "correlation_matrix"   : pd.DataFrame
          - "vif_table"            : Dict[str, float] or pd.Series
          - "singular_values"      : List[float]
          - "condition_number"     : float
    """
    dropped_features: List[str]
    notes: List[str]
    extra: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Ensure extra is always a dictionary to simplify callers.
        if self.extra is None:
            self.extra = {}


# =============================================================================
# Base class for all reducers
# =============================================================================

class BaseCollinearityReducer(ABC):
    """
    Abstract base class for all collinearity reduction strategies.

    Each reducer must implement the method:
        fit_transform(X, y=None) -> (X_reduced, diagnostics)

    Contract
    --------
    - Input:
        X : pd.DataFrame
            Numeric feature matrix. Column names must be preserved.
        y : optional target vector (ignored by some strategies).

    - Output:
        X_reduced : pd.DataFrame
            Possibly-modified DataFrame. Column order should be preserved for
            remaining columns.

        diagnostics : CollinearityDiagnostics
            Information about what was changed and why.

    Reducers MUST NOT:
    ------------------
    - Mutate the input DataFrame X in-place.
    - Assume a specific model (OLS, Ridge, etc.) is being used.
    """

    # Short name used in logs and in the registry
    name: str = "base"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        config : Dict[str, Any] or None
            Strategy-specific configuration, typically loaded from a model
            config YAML (e.g., VIF threshold, Pearson correlation threshold).
        """
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> Tuple[pd.DataFrame, CollinearityDiagnostics]:
        """
        Perform detection and/or reduction on current X.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (numeric). Index and column names should be stable.

        y : Any, optional
            Target values. Some strategies might use y in the future (e.g.,
            supervised feature selection), but current ones ignore it.

        Returns
        -------
        X_reduced : pd.DataFrame
            DataFrame with some features possibly removed.

        diagnostics : CollinearityDiagnostics
            Structured metadata describing feature drops and metrics.
        """
        raise NotImplementedError("Subclasses must implement fit_transform().")


# =============================================================================
# Pearson correlation-based reducer
# =============================================================================

class PearsonReducer(BaseCollinearityReducer):
    """
    Removes highly correlated feature pairs using the Pearson coefficient.

    Approach
    --------
    - Compute the full pairwise correlation matrix of X.
    - Consider only the upper triangle (i < j) to avoid duplicate pairs.
    - For every pair (col_i, col_j) where |corr| >= abs_threshold:
        * Drop one of the two features according to the chosen policy.

    Configuration (from self.config)
    --------------------------------
    abs_threshold : float, default 0.95
        Absolute correlation threshold above which we consider a pair
        "too correlated".

    drop_policy : {"second", "first"}, default "second"
        Strategy for which feature in a correlated pair to drop:
        - "second" : always drop the second feature in the pair.
                     This is deterministic if column order is fixed.
        - "first"  : always drop the first feature in the pair.

        In more advanced versions you could support:
        - "keep_longer_name", "keep_higher_importance", etc.

    Notes
    -----
    - This method detects direct (pairwise) redundancy only.
    - It does NOT capture multivariate collinearity involving 3+ features.
    """

    name = "pearson"

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None
    ) -> Tuple[pd.DataFrame, CollinearityDiagnostics]:
        # Read configuration with sensible defaults.
        threshold = float(self.config.get("abs_threshold", 0.95))
        drop_policy = self.config.get("drop_policy", "second")  # "first" | "second"

        notes: List[str] = []
        to_drop: set[str] = set()

        # Compute absolute correlation matrix once.
        # We keep the full matrix for logging and diagnostics.
        corr = X.corr(method="pearson").abs()
        cols = corr.columns.tolist()

        # Traverse upper triangle of the correlation matrix.
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                value = corr.iloc[i, j]

                if value >= threshold:
                    # Decide which feature to drop based on drop_policy.
                    if drop_policy == "first":
                        dropped_feature = c1
                    else:
                        dropped_feature = c2

                    # Mark feature for removal but do not drop yet.
                    to_drop.add(dropped_feature)

                    notes.append(
                        f"Pearson | {c1}–{c2} | corr={value:.3f} "
                        f"≥ {threshold:.3f} → drop {dropped_feature}"
                    )

        # Actually drop all selected features.
        dropped = list(to_drop)
        X_new = X.drop(columns=dropped) if dropped else X.copy()

        diagnostics = CollinearityDiagnostics(
            dropped_features=dropped,
            notes=notes,
            extra={
                # Full correlation matrix can be logged as CSV or heatmap.
                "correlation_matrix": corr,
            },
        )
        return X_new, diagnostics


# =============================================================================
# VIF-based reducer (Variance Inflation Factor)
# =============================================================================

class VIFReducer(BaseCollinearityReducer):
    """
    Removes features based on Variance Inflation Factor (VIF).

    Background
    ----------
    For each feature X_j:

        1. Regress X_j on all other features X_-j via OLS.
        2. Let R_j^2 be the coefficient of determination from that regression.
        3. Then VIF_j = 1 / (1 - R_j^2).

    Interpretation:
        - VIF_j ≈ 1   : no collinearity.
        - VIF_j > 5   : moderate collinearity (rule-of-thumb).
        - VIF_j > 10  : severe collinearity (classical cutoff).

    Algorithm
    ---------
    1. Repeatedly:
        a) Compute VIF for all current features.
        b) If max VIF <= threshold: stop.
        c) Else drop the feature with the largest VIF.
    2. Stop if max_iter iterations are reached (safety measure).

    Configuration (from self.config)
    --------------------------------
    threshold : float, default 10.0
        VIF threshold above which a feature is considered problematic.

    max_iter : int, default 10
        Maximum number of features to drop. Prevents infinite loops in
        degenerate cases.

    Notes
    -----
    - Computationally O(K * N * K) where K is number of features,
      since for each feature we fit a regression against K-1 others.
      Thus best for moderate-dimensional problems (e.g., K < 50–100).
    """

    name = "vif"

    def _compute_vif(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute VIF for each feature in X.

        Parameters
        ----------
        X : pd.DataFrame
            Numeric features.

        Returns
        -------
        vif_series : pd.Series
            Index = feature name, value = VIF.
        """
        vifs: Dict[str, float] = {}

        for feature in X.columns:
            # Target is the current feature.
            y = X[feature].values

            # Predictors are all other features.
            X_rest = X.drop(columns=[feature]).values

            # Simple OLS using sklearn's LinearRegression.
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_rest, y)

            # R^2 for regression of feature on the rest.
            r2 = lr.score(X_rest, y)

            # Add a tiny epsilon to avoid division by zero when r2 ≈ 1.0.
            vif = 1.0 / (1.0 - r2 + 1e-12)
            vifs[feature] = float(vif)

        return pd.Series(vifs)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None
    ) -> Tuple[pd.DataFrame, CollinearityDiagnostics]:
        threshold = float(self.config.get("threshold", 10.0))
        max_iter = int(self.config.get("max_iter", 10))

        X_cur = X.copy()
        notes: List[str] = []
        dropped: List[str] = []

        # Iteratively drop worst VIF feature until all are below threshold
        # or max_iter is reached.
        for iteration in range(max_iter):
            # If only one feature remains, we cannot compute VIF anymore.
            if X_cur.shape[1] <= 1:
                notes.append(
                    "VIF | stopping because only one feature remains; "
                    "no further VIF computation is meaningful."
                )
                break

            vifs = self._compute_vif(X_cur)
            worst_feature = vifs.idxmax()
            worst_value = float(vifs.max())

            if worst_value <= threshold:
                notes.append(
                    f"VIF | all features have VIF <= {threshold:.2f}; stopping."
                )
                break

            # Drop the feature with the highest VIF.
            dropped.append(worst_feature)
            X_cur = X_cur.drop(columns=[worst_feature])
            notes.append(
                f"VIF | iteration {iteration + 1} | dropped '{worst_feature}' "
                f"(VIF={worst_value:.2f})"
            )

        # We log the last computed VIF table (vifs) if it exists.
        extra: Dict[str, Any] = {}
        try:
            extra["vif_table"] = vifs.to_dict()  # type: ignore[name-defined]
        except Exception:
            # In edge cases (e.g., no iteration run), vifs may not exist.
            extra["vif_table"] = {}

        diagnostics = CollinearityDiagnostics(
            dropped_features=dropped,
            notes=notes,
            extra=extra,
        )
        return X_cur, diagnostics


# =============================================================================
# SVD-based diagnostics (usually no feature removal)
# =============================================================================

class SVDReducer(BaseCollinearityReducer):
    """
    Performs SVD on a standardized version of X to examine singular values.

    Purpose
    -------
    - Identify near-rank-deficient design matrices.
    - Provide a numerically stable way to inspect the "shape" of X.

    Procedure
    ---------
    1. Standardize columns of X (mean 0, std 1) to remove scale effects.
    2. Compute SVD: X_std = U * diag(s) * V^T.
    3. Condition number kappa = s_max / s_min:
        - Large kappa → small singular values → near-dependencies.

    Configuration (from self.config)
    --------------------------------
    action : {"warn"}, default "warn"
        Current implementation is diagnostic-only, so "warn" is the
        only supported behavior. In the future, "drop" could be used
        for aggressive dimensionality reduction.

    tiny_singular_value_threshold : float, default 1e-6
        Values below this may be considered "numerically zero".

    Notes
    -----
    - This reducer does not (yet) drop any features; X is returned unchanged.
    - The diagnostics can be used by higher-level code to trigger
      alternative models (e.g., switch from OLS to Ridge).
    """

    name = "svd"

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None
    ) -> Tuple[pd.DataFrame, CollinearityDiagnostics]:
        action = self.config.get("action", "warn")
        tiny_thresh = float(self.config.get("tiny_singular_value_threshold", 1e-6))

        # Standardize features to put them on comparable scale.
        # Add epsilon to std to avoid division by zero.
        X_std = (X - X.mean()) / (X.std() + 1e-12)

        # Compute singular values. full_matrices=False is sufficient for diagnostics.
        _, singular_values, _ = np.linalg.svd(X_std.values, full_matrices=False)
        s = singular_values
        cond_number = float(s[0] / (s[-1] + 1e-12))

        notes: List[str] = [
            f"SVD | singular values={s.tolist()}",
            f"SVD | condition number={cond_number:.2f}",
        ]

        # Provide a quick summary about "tiny" singular values.
        tiny_count = int(np.sum(s < tiny_thresh))
        if tiny_count > 0:
            notes.append(
                f"SVD | {tiny_count} singular values < {tiny_thresh:.1e} "
                "(indicates near-linear dependencies)."
            )

        if action == "warn":
            # Diagnostic-only mode: we do not modify X.
            pass
        else:
            # Future extension: "drop" mode could project onto largest singular
            # vectors or attempt to identify columns responsible for near-zero
            # singular values.
            notes.append(
                f"SVD | unsupported action '{action}' → no features dropped."
            )

        diagnostics = CollinearityDiagnostics(
            dropped_features=[],
            notes=notes,
            extra={
                "singular_values": s.tolist(),
                "condition_number": cond_number,
            },
        )
        return X, diagnostics


# =============================================================================
# Condition number diagnostics (fast, scalar indicator)
# =============================================================================

class ConditionNumberReducer(BaseCollinearityReducer):
    """
    Computes a condition number as a coarse stability indicator.

    Interpretation
    --------------
    - The condition number captures how sensitive solutions of linear systems
      are to perturbations in the data. For OLS, large condition numbers
      imply unstable coefficient estimates.

    Implementation
    --------------
    - We call numpy.linalg.cond(X) directly on the design matrix X.
      This uses the default 2-norm (based on singular values).

    Configuration (from self.config)
    --------------------------------
    threshold : float, default 30.0
        Values above this threshold are considered "problematic" and will
        be flagged in the diagnostics notes.

    Notes
    -----
    - This reducer does not modify X; it is diagnostic-only.
    - For more detailed information, use the SVDReducer.
    """

    name = "condition_number"

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None
    ) -> Tuple[pd.DataFrame, CollinearityDiagnostics]:
        threshold = float(self.config.get("threshold", 30.0))

        # Compute condition number on raw X.
        cond_value = float(np.linalg.cond(X.values))

        notes: List[str] = [
            f"ConditionNumber | cond={cond_value:.2f}",
        ]

        if cond_value > threshold:
            notes.append(
                f"ConditionNumber | WARNING: cond={cond_value:.2f} exceeds "
                f"threshold={threshold:.2f} → potential multicollinearity."
            )

        diagnostics = CollinearityDiagnostics(
            dropped_features=[],
            notes=notes,
            extra={"condition_number": cond_value},
        )
        return X, diagnostics


# =============================================================================
# Registry of available reducers
# =============================================================================

REDUCER_REGISTRY: Dict[str, type[BaseCollinearityReducer]] = {
    "pearson": PearsonReducer,
    "vif": VIFReducer,
    "svd": SVDReducer,
    "condition_number": ConditionNumberReducer,
}
