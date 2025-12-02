"""
===============================================================================
  CollinearityManager — Orchestrates Collinearity Strategies
===============================================================================

This module defines CollinearityManager, a small orchestration layer that:
  - reads a collinearity configuration (typically from a model YAML),
  - instantiates one or more reducers (Pearson, VIF, SVD, etc.),
  - applies them in sequence to the design matrix X,
  - stores structured diagnostics for logging and analysis,
  - remembers which columns were dropped so it can transform new data
    (e.g., validation / test sets) in a consistent way.

The manager itself is:
  - model-agnostic (it never touches the estimator),
  - dataset-agnostic (it only sees the feature matrix X, y),
  - driven entirely by configuration.

Example config (from a model YAML)
----------------------------------

collinearity:
  enabled: true

  # How to choose which strategies to run:
  mode: "manual"   # or "auto"

  # In manual mode, run these strategies in order:
  pipeline: ["pearson", "vif"]

  pearson:
    abs_threshold: 0.95
    drop_policy: "second"

  vif:
    threshold: 10.0
    max_iter: 10

  svd:
    action: "warn"
    tiny_singular_value_threshold: 1e-6

  condition_number:
    threshold: 30.0

  # In auto mode, the manager uses cheap heuristics based on this policy:
  auto_policy:
    objective: "interpretability"   # or "stability"
    max_features_for_vif: 50
    use_svd: true
    use_condition_number: true

Training pipeline usage
-----------------------

from src.features.collinearity_manager import CollinearityManager

manager = CollinearityManager(config["collinearity"])
X_train_col = manager.fit_transform(X_train, y_train)
X_val_col   = manager.transform(X_val)  # drops the same columns

diagnostics = manager.get_all_diagnostics()
===============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from .collinearity_strategies import (
    BaseCollinearityReducer,
    CollinearityDiagnostics,
    REDUCER_REGISTRY,
)


class CollinearityManager:
    """
    Orchestrates one or more collinearity reduction strategies.

    Responsibilities
    ----------------
    - Interpret the high-level collinearity configuration.
    - Build an ordered list of reducer objects (Pearson, VIF, etc.).
    - Apply reducers sequentially to X during training (`fit_transform`).
    - Record which columns were dropped.
    - Provide a `transform` method to apply the same column dropping to
      new data (validation, test) without recomputing diagnostics.

    Configuration contract (expected keys in config dict)
    -----------------------------------------------------
    enabled : bool
        If False, the manager becomes a no-op and simply returns X unchanged.

    mode : {"manual", "auto"}, default "manual"
        - "manual": use the explicit 'pipeline' list.
        - "auto"  : choose a strategy pipeline based on a simple heuristic.

    pipeline : List[str]
        List of reducer names to apply in sequence (only in manual mode).
        Example: ["pearson", "vif"]

    pearson, vif, svd, condition_number : Dict[str, Any]
        Sub-configs passed directly to the corresponding reducer classes.

    auto_policy : Dict[str, Any]
        Extra hints for automatic pipeline selection, e.g.:
          objective: "interpretability" or "stability"
          max_features_for_vif: int
          use_svd: bool
          use_condition_number: bool

    Notes
    -----
    - The manager does not validate that config keys are present; it uses
      sensible defaults and ignores unknown keys.
    - All heavy numerical work is done by the reducers themselves.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """
        Initialize the manager with a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any] or None
            Full collinearity configuration section for a model. This is
            usually loaded from a model-level YAML file.
        """
        self.config: Dict[str, Any] = config or {}

        # Indicates whether fit_transform() has been called successfully.
        self._fitted: bool = False

        # Original feature names seen during fit_transform().
        self.feature_names_in_: List[str] = []

        # Feature names after all reducers have been applied.
        self.feature_names_out_: List[str] = []

        # Aggregate list of columns dropped by all reducers.
        self._dropped_features: List[str] = []

        # Diagnostics from each individual reducer, in execution order.
        self._all_diagnostics: List[CollinearityDiagnostics] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> pd.DataFrame:
        """
        Fit the collinearity pipeline on X (and optional y) and return
        a reduced version of X with problematic columns removed.

        This method:
          - Reads the configuration (enabled + mode).
          - Builds an ordered list of reducers.
          - Applies each reducer sequentially.
          - Stores which columns were dropped for later use in transform().

        Parameters
        ----------
        X : pd.DataFrame
            Training features after general feature engineering (e.g.,
            datetime/spatial transforms). Must have stable column names.

        y : Any, optional
            Training targets. Some reducers (currently none) might make use
            of y in the future.

        Returns
        -------
        X_reduced : pd.DataFrame
            DataFrame with the same index as X, but possibly fewer columns.
        """
        if not self.config.get("enabled", False):
            # Manager is disabled: do nothing.
            self._fitted = True
            self.feature_names_in_ = list(X.columns)
            self.feature_names_out_ = list(X.columns)
            self._dropped_features = []
            self._all_diagnostics = []
            return X.copy()

        self.feature_names_in_ = list(X.columns)

        # Build the sequence of reducers based on mode and configuration.
        mode = self.config.get("mode", "manual")
        if mode == "auto":
            reducers = self._build_auto_pipeline(X)
        else:
            reducers = self._build_manual_pipeline()

        X_cur = X.copy()
        all_diagnostics: List[CollinearityDiagnostics] = []
        dropped_features: List[str] = []

        # Apply reducers in order.
        for reducer in reducers:
            X_cur, diagnostics = reducer.fit_transform(X_cur, y)
            all_diagnostics.append(diagnostics)
            dropped_features.extend(diagnostics.dropped_features)

        # Record final state.
        self._fitted = True
        self._all_diagnostics = all_diagnostics
        self._dropped_features = dropped_features
        self.feature_names_out_ = list(X_cur.columns)

        return X_cur

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned column dropping to a new dataset (e.g. validation).

        This method does NOT recompute diagnostics; it simply ensures that
        the output DataFrame has the same columns as the reduced training
        DataFrame returned by fit_transform().

        Parameters
        ----------
        X : pd.DataFrame
            New data with at least the columns present during training.

        Returns
        -------
        X_aligned : pd.DataFrame
            DataFrame restricted to the columns kept during training.
            Columns are ordered identically to feature_names_out_.
        """
        if not self._fitted:
            raise RuntimeError(
                "CollinearityManager.transform() called before fit_transform(). "
                "You must fit the manager on training data first."
            )

        # We keep only columns learned during fit_transform().
        missing_cols = [c for c in self.feature_names_out_ if c not in X.columns]
        if missing_cols:
            # This is not fatal; we simply drop those missing columns.
            # But we log an explicit warning message via exception text.
            # In production you may want to use the logging module instead.
            raise ValueError(
                "CollinearityManager.transform(): input DataFrame is missing "
                f"the following columns expected after training: {missing_cols}"
            )

        # Reindex to match the learned column order exactly.
        X_aligned = X[self.feature_names_out_].copy()
        return X_aligned

    def get_all_diagnostics(self) -> List[CollinearityDiagnostics]:
        """
        Return diagnostics from all reducers, in the order they were applied.

        Each CollinearityDiagnostics object contains:
          - dropped_features : List[str]
          - notes            : List[str]
          - extra            : Dict[str, Any]

        Returns
        -------
        diagnostics_list : List[CollinearityDiagnostics]
            One entry per reducer that ran during fit_transform().
        """
        if not self._fitted:
            raise RuntimeError(
                "CollinearityManager.get_all_diagnostics() called before "
                "fit_transform()."
            )
        return list(self._all_diagnostics)

    def get_dropped_features(self) -> List[str]:
        """
        Return the list of all features dropped by all reducers.

        Returns
        -------
        dropped : List[str]
            Names of features removed during fit_transform().
        """
        if not self._fitted:
            raise RuntimeError(
                "CollinearityManager.get_dropped_features() called before "
                "fit_transform()."
            )
        return list(self._dropped_features)

    # -------------------------------------------------------------------------
    # Internal helpers: pipeline construction
    # -------------------------------------------------------------------------

    def _build_manual_pipeline(self) -> List[BaseCollinearityReducer]:
        """
        Build an explicit sequence of reducers based on the 'pipeline' list
        in the configuration.

        Example:
            pipeline: ["pearson", "vif"]

        For each name, we:
          1. Look up the corresponding class in REDUCER_REGISTRY.
          2. Instantiate it with its sub-config, e.g. config["pearson"].

        Returns
        -------
        reducers : List[BaseCollinearityReducer]
            List of configured reducer instances.
        """
        methods: Sequence[str] = self.config.get("pipeline", [])
        reducers: List[BaseCollinearityReducer] = []

        for method_name in methods:
            cls = REDUCER_REGISTRY.get(method_name)
            if cls is None:
                # Ignore unknown method names, but you could also raise
                # an exception here if you prefer strict behavior.
                continue

            sub_config = self.config.get(method_name, {})
            reducer = cls(sub_config)
            reducers.append(reducer)

        return reducers

    def _build_auto_pipeline(self, X: pd.DataFrame) -> List[BaseCollinearityReducer]:
        """
        Automatically build a pipeline of reducers based on simple heuristics.

        Rationale
        ---------
        - For OLS, we usually want at least Pearson + VIF.
        - For high-dimensional X, VIF may be too expensive, so we may skip it.
        - For stability-focused runs, we can add purely diagnostic SVD and
          condition number checks.

        Behavior (simple heuristic)
        ---------------------------
        1. Always include Pearson if available.
        2. If n_features <= max_features_for_vif: also include VIF.
        3. If objective == "stability" and the relevant flags are True:
             optionally add SVD and/or condition number at the end.

        Parameters
        ----------
        X : pd.DataFrame
            Training design matrix. Only used to inspect feature count.

        Returns
        -------
        reducers : List[BaseCollinearityReducer]
            List of reducer instances built according to auto policy.
        """
        policy: Dict[str, Any] = self.config.get("auto_policy", {})
        objective: str = policy.get("objective", "interpretability")
        max_features_for_vif: int = int(policy.get("max_features_for_vif", 50))
        use_svd: bool = bool(policy.get("use_svd", True))
        use_cond: bool = bool(policy.get("use_condition_number", True))

        n_features = X.shape[1]
        reducers: List[BaseCollinearityReducer] = []

        # 1) Pearson is cheap and generally useful for a first pass.
        if "pearson" in REDUCER_REGISTRY:
            pearson_cfg = self.config.get("pearson", {})
            reducers.append(REDUCER_REGISTRY["pearson"](pearson_cfg))

        # 2) VIF only if dimensionality is manageable.
        if n_features <= max_features_for_vif and "vif" in REDUCER_REGISTRY:
            vif_cfg = self.config.get("vif", {})
            reducers.append(REDUCER_REGISTRY["vif"](vif_cfg))

        # 3) Stability-oriented diagnostics: SVD + condition number.
        if objective == "stability":
            if use_svd and "svd" in REDUCER_REGISTRY:
                svd_cfg = self.config.get("svd", {})
                reducers.append(REDUCER_REGISTRY["svd"](svd_cfg))

            if use_cond and "condition_number" in REDUCER_REGISTRY:
                cond_cfg = self.config.get("condition_number", {})
                reducers.append(REDUCER_REGISTRY["condition_number"](cond_cfg))

        return reducers
