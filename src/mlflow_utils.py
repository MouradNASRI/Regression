"""
Centralized MLflow utilities:
 - Load MLflow config from configs/mlflow/<name>.yml
 - Initialize MLflow tracking/experiment/autolog behavior
"""
from typing import Any, Dict, List

import pandas as pd
from pathlib import Path
import yaml
import mlflow
import os
from sklearn.pipeline import Pipeline

from src.features.collinearity_manager import CollinearityManager
from src.features.collinearity_strategies import CollinearityDiagnostics

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "mlflow"


def load_mlflow_config(name: str = "local") -> dict:
    """
    Loads MLflow configuration from configs/mlflow/<name>.yml

    Example folder structure:
        configs/mlflow/local.yml
        configs/mlflow/staging.yml
    """
    cfg_path = CONFIG_ROOT / f"{name}.yml"

    if not cfg_path.exists():
        raise ValueError(f"❌ MLflow profile '{name}' not found at {cfg_path}")

    return yaml.safe_load(cfg_path.read_text())


def init_mlflow_tracking(cfg: dict):
    """
    Applies MLflow configuration loaded via load_mlflow_config()

    Expected keys:
      - tracking_uri
      - registry_uri (optional)
      - experiment
      - autolog (bool)

    This MUST be called BEFORE model.fit()
    """

    if cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(cfg["tracking_uri"])

    if cfg.get("registry_uri"):
        mlflow.set_registry_uri(cfg["registry_uri"])

    if cfg.get("experiment"):
        mlflow.set_experiment(cfg["experiment"])

    # Optional autolog
    if cfg.get("autolog", False):
        mlflow.sklearn.autolog(log_models=False)  # we manually log models later

    print(f"🔧 MLflow configured: {cfg}")

def load_and_init_mlflow(profile: str, 
                         cli_experiment: str | None = None):
    """
    Loads MLflow config YAML, applies precedence rules, initializes MLflow tracking.
    
    Precedence:
    1. CLI argument (--experiment)
    2. YAML config experiment
    3. Env var MLFLOW_EXPERIMENT_NAME
    4. No experiment (MLflow default)
    """

    cfg = load_mlflow_config(profile)

    # Apply precedence logic HERE (not in main)
    experiment = (
        cli_experiment
        or cfg.get("experiment")
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
    )
    cfg["experiment"] = experiment

    init_mlflow_tracking(cfg)
    return cfg


def log_with_mlflow(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    pre=None,
    experiment: str | None = None,
    autolog_on=False,
    col_manager: CollinearityManager | None = None,
    model_mlflow_cfg: dict | None = None,
):
    """
    Wrapper that:
      - starts a run (or nested run if one is active)
      - logs params and metrics
      - logs model or (preprocessor+model) pipeline
    """
    active = mlflow.active_run()
    started_here = False

    if active is None:
        if experiment:
            mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        started_here = True
    else:
        mlflow.start_run(run_name=run_name, nested=True)

    try:
        mlflow.set_tag("model_name", run_name)
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        # NEW: log collinearity diagnostics if a manager was provided.
        if col_manager is not None:
            log_collinearity_diagnostics_to_mlflow(
                col_manager=col_manager,
                model_mlflow_cfg=model_mlflow_cfg,
            )


        # Avoid double-logging if autolog is managing models
        if not autolog_on:
            try:
                if pre is not None:
                    pipeline = Pipeline([("preprocessor", pre), ("model", model)])
                    mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception:
                pass
            
    finally:
        mlflow.end_run()
        if started_here:
            # no outer run to restore
            pass


def log_collinearity_diagnostics_to_mlflow(
    col_manager: CollinearityManager,
    model_mlflow_cfg: Dict[str, Any] | None = None,
) -> None:
    """
    Logs collinearity reduction diagnostics (Pearson, VIF, SVD, etc.)
    to the *currently active* MLflow run.

    This function expects that:
      - col_manager has already been fitted via fit_transform() on training data.
      - an MLflow run is active (started in log_with_mlflow()).

    Parameters
    ----------
    col_manager : CollinearityManager
        Fitted manager that exposes:
          - get_all_diagnostics()
          - get_dropped_features()
          - feature_names_out_

    model_mlflow_cfg : dict, optional
        The `mlflow` subsection of the model config (e.g. configs/models/ols.yaml),
        used to control what to log. Expected optional keys:

            log_collinearity_diagnostics : bool
            log_vif_table               : bool
            log_correlation_heatmap     : bool (reserved for future use)
            log_svd_spectrum            : bool

        If None, we assume diagnostics logging is enabled.
    """
    # Normalize config and check if user explicitly disabled diagnostics.
    model_mlflow_cfg = model_mlflow_cfg or {}
    if not model_mlflow_cfg.get("log_collinearity_diagnostics", True):
        # User chose not to log collinearity details.
        return

    # -------------------------------------------------------------------------
    # 1) Global summary: dropped features and final retained features
    # -------------------------------------------------------------------------
    all_dropped: List[str] = col_manager.get_dropped_features()
    final_features: List[str] = col_manager.feature_names_out_

    # Log summary information as MLflow params.
    mlflow.log_param(
        "collinearity_dropped_features",
        ",".join(all_dropped) if all_dropped else "",
    )
    mlflow.log_param(
        "collinearity_final_features",
        ",".join(final_features),
    )
    mlflow.log_param(
        "collinearity_final_feature_count",
        len(final_features),
    )

    # -------------------------------------------------------------------------
    # 2) Per-reducer diagnostics (Pearson, VIF, SVD, condition number, etc.)
    # -------------------------------------------------------------------------
    diagnostics_list: List[CollinearityDiagnostics] = list(
        col_manager.get_all_diagnostics()
    )

    for idx, diag in enumerate(diagnostics_list):
        prefix = f"col_{idx}"

        # Log which features this particular reducer dropped.
        if diag.dropped_features:
            mlflow.log_param(
                f"{prefix}_dropped_features",
                ",".join(diag.dropped_features),
            )

        # Log short human-readable notes for this reducer.
        if diag.notes:
            mlflow.log_param(
                f"{prefix}_notes",
                " | ".join(diag.notes),
            )

        extra = diag.extra or {}

        # ---------------------------------------------------------------------
        # 2a) VIF table (if present and user wants it)
        # ---------------------------------------------------------------------
        if model_mlflow_cfg.get("log_vif_table", True):
            vif_table = extra.get("vif_table")
            if isinstance(vif_table, dict) and vif_table:
                vif_df = pd.DataFrame(
                    {
                        "feature": list(vif_table.keys()),
                        "vif": list(vif_table.values()),
                    }
                )
                vif_path = f"collinearity/{prefix}_vif_table.csv"
                vif_df.to_csv(vif_path, index=False)
                mlflow.log_artifact(vif_path)

        # ---------------------------------------------------------------------
        # 2b) Correlation matrix (from PearsonReducer)
        # ---------------------------------------------------------------------
        corr = extra.get("correlation_matrix")
        if isinstance(corr, pd.DataFrame):
            corr_path = f"collinearity/{prefix}_correlation_matrix.csv"
            corr.to_csv(corr_path)
            mlflow.log_artifact(corr_path)

        # ---------------------------------------------------------------------
        # 2c) SVD singular values and condition number
        # ---------------------------------------------------------------------
        if model_mlflow_cfg.get("log_svd_spectrum", False):
            singular_values = extra.get("singular_values")
            if isinstance(singular_values, list):
                svd_path = f"collinearity/{prefix}_singular_values.csv"
                pd.DataFrame({"singular_value": singular_values}).to_csv(
                    svd_path, index=False
                )
                mlflow.log_artifact(svd_path)

        cond = extra.get("condition_number")
        if cond is not None:
            mlflow.log_metric(f"{prefix}_condition_number", float(cond))
