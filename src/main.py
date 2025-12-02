"""
Main entry point for training / evaluation / hyperparameter search.

This version integrates:
  - dataset config  (configs/datasets/<name>.yaml)
  - model config    (configs/models/<model>.yaml)
  - optional collinearity handling driven by model config["collinearity"]

Collinearity handling is:
  - applied AFTER load_data (i.e. after your feature engineering),
  - applied BEFORE train_eval_log / run_hparam_search,
  - controlled entirely via configs/models/<model>.yaml.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.cli_utils import build_parser, parse_kv, list_available_models
from src.data_utils import load_data, load_dataset_config
from src.mlflow_utils import load_and_init_mlflow
from src.training_utils import train_eval_log
from src.hparam_search import load_hparam_search_config, run_hparam_search
from src.features.collinearity_manager import CollinearityManager
from src.model_utils import load_model_config

def main() -> None:
    """
    Main orchestration function.

    Responsibilities
    ----------------
    1) Parse CLI arguments.
    2) Initialize MLflow according to profile.
    3) Load dataset + model configs.
    4) Load and preprocess data (including your existing feature engineering).
    5) Optionally apply collinearity reduction based on model config.
    6) Run either hyperparameter search or a single train/eval/log run.
    """
    #load and parse the CLI arguments
    parser = build_parser()
    args = parser.parse_args()
    
    # Load mlflow config file and init mlflow cfg from configs/mlflow/<profile>.yml
    mlflow_cfg = load_and_init_mlflow(args.mlflow_profile, 
                                      cli_experiment=args.experiment)

    #Load dataset cconfig from config/datasets/<name>.yaml
    dataset_cfg = load_dataset_config(args.dataset)

    if args.list_models:
        list_available_models()

    # -------------------------------------------------------------------------
    # 4) Load model config from configs/models/<model>.yaml
    # -------------------------------------------------------------------------
    # The model config includes:
    #   - model: name, library, type, params
    #   - collinearity: enabled/mode/pipeline/pearson/vif/svd/condition_number
    #   - (optionally) mlflow settings specific to the model
    model_cfg: Dict[str, Any] = load_model_config(args.model)

        #  Decide whether to scale numeric features
    dataset_scale_default = dataset_cfg.get("scale_numeric", True)
    model_pre_cfg = model_cfg.get("preprocessing", {})

    if "scale_numeric" in model_pre_cfg:
        # Model overrides dataset default
        scale_numeric = model_pre_cfg["scale_numeric"]
    else:
        # Fall back to dataset’s default behaviour
        scale_numeric = dataset_scale_default


    

    # -------------------------------------------------------------------------
    # 5) Load & preprocess data (your existing feature engineering)
    # -------------------------------------------------------------------------
    # NOTE:
    #   - X_train, X_test : numpy arrays after preprocessing
    #   - y_train, y_test : target vectors
    #   - pre             : preprocessor (for logging or reuse)
    #   - feat_names      : list[str] of feature names used to build X arrays
    #   - X_train_df, X_test_df : raw DataFrames (unused here but kept)
    X_train, X_test, y_train, y_test, pre, feat_names, X_train_df, X_test_df = load_data(
        path=args.data,
        target=dataset_cfg["target"],
        test_size=args.test_train_ratio,
        random_state=42,
        drop_cols=dataset_cfg.get("drop_cols", []),
        keep_categoricals=dataset_cfg.get("keep_categoricals", False),
        scale_numeric=scale_numeric,
        time_split=dataset_cfg.get("time_split", False),
        datetime_col=dataset_cfg.get("datetime_col"),
        feature_engineering=dataset_cfg.get("feature_engineering", {}),
        spatial_cols=dataset_cfg.get("spatial_cols"),  # only for taxi-like datasets
        rename_cols=dataset_cfg.get("rename_cols", {}),
        return_preprocessor=True,
        return_raw=True
    )

    # -------------------------------------------------------------------------
    # 6) Parse CLI model params (e.g. --model_params alpha=0.1)
    # -------------------------------------------------------------------------
    cli_model_params: Dict[str, Any] = parse_kv(args.model_params)

    # -------------------------------------------------------------------------
    # 7) Apply collinearity reduction, if enabled in model config
    # -------------------------------------------------------------------------
    # This step is model-aware and config-driven:
    #   - If model_cfg["collinearity"]["enabled"] is True:
    #       * build a CollinearityManager with that config
    #       * run fit_transform() on training features
    #       * run transform() on test features
    #   - If disabled or missing, we leave X_train / X_test unchanged.
    #
    # IMPORTANT:
    #   CollinearityManager expects pandas.DataFrame with column names,
    #   so we wrap the arrays back to DataFrames using feat_names, then
    #   convert back to numpy after reduction.
    col_cfg: Dict[str, Any] = model_cfg.get("collinearity", {})
    model_mlflow_cfg = model_cfg.get("mlflow", {})
    col_manager = None

    if col_cfg.get("enabled", False):
        # Wrap arrays into DataFrames with feature names.
        X_train_df_col = pd.DataFrame(X_train, columns=feat_names)
        X_test_df_col = pd.DataFrame(X_test, columns=feat_names)

        # Initialize manager with model-level collinearity configuration.
        col_manager = CollinearityManager(col_cfg)

        # Fit on training data and reduce training feature matrix.
        X_train_df_col_reduced = col_manager.fit_transform(X_train_df_col, y_train)

        # Apply the same column selection to test data.
        X_test_df_col_reduced = col_manager.transform(X_test_df_col)

        # Convert back to numpy arrays for downstream code.
        X_train = X_train_df_col_reduced.values
        X_test = X_test_df_col_reduced.values
        feat_names = list(X_train_df_col_reduced.columns)

        # Optional: print or log which features were dropped.
        dropped_features = col_manager.get_dropped_features()
        print("[Collinearity] Dropped features from model config:", dropped_features)
    else:
        # Collinearity handling is disabled for this model.
        print("[Collinearity] Disabled in model config; skipping collinearity reduction.")

    # -------------------------------------------------------------------------
    # 8) Either run hyperparameter search or a single training run
    # -------------------------------------------------------------------------
    if args.hparam_search:
        search_cfg = load_hparam_search_config(name = args.hparam_search, model_name=args.model,)
        cli_model_name=args.model
        mlflow_experiment=args.experiment

        results = run_hparam_search(
            search_cfg=search_cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pre=pre,
            cli_model_params=cli_model_params,
            cli_model_name=cli_model_name,
            mlflow_experiment=mlflow_experiment,
        )

        print(
            f"[{results['search_name']}] "
            f"Best {results['metric_name']} = {results['best_metric_value']:.4f}"
        )
        print("Best params:")
        for k, v in results["best_params"].items():
            print(f"  {k} = {v}")

    else:
        # Single run (your old behaviour)
        train_eval_log(
            model_name=args.model,
            model_params=cli_model_params or {},
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pre=pre,
            mlflow_experiment=mlflow_experiment,
            col_manager=col_manager,          # NEW
            model_mlflow_cfg=model_mlflow_cfg # NEW
        )

if __name__ == "__main__":
    main()

