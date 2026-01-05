# src/training_utils.py

from typing import Dict, Any
from sklearn.base import BaseEstimator
from src.model_registry import get_model as get_task_model
from src.mlflow_utils import log_with_mlflow
from src.tasks.registry import get_task_evaluator
from src.factories.spec_builder import build_from_spec


def train_eval_log(
    model_name: str,
    model_params: Dict,
    X_train,
    y_train,
    X_test,
    y_test,
    pre=None,
    mlflow_experiment: str | None = None,
    col_manager=None,
    model_mlflow_cfg: dict | None = None,
    task: str = "regression",
) -> Dict[str, Any]:
    """
    Generic training routine for sklearn-compatible models.
    Performs:
      - model construction (via registry)
      - fit() on train data
      - predict() on test data
      - MSE + R² evaluation
      - MLflow logging (with optional preprocessor pipeline)
      - task : str
        The type of ML task (e.g. "regression", "classification").
        This controls which metrics are computed.

    Returns:
      metrics: Dict[str, float]
    """

    # ------------------------------------------------------------
    # 1️⃣ BUILD MODEL
    # ------------------------------------------------------------
    # model : BaseEstimator = get_task_model(task, model_name, **(model_params or {}))
    
    # Pattern A: spec-driven construction
    # model_name must exist in ESTIMATOR_CATALOG (as a "type" key)
    spec = {
        "type": model_name,
        "params": (model_params or {}),
    }
    model: BaseEstimator = build_from_spec(spec)

    # Capability check
    has_fit     = hasattr(model, "fit")
    has_predict = hasattr(model, "predict")

    print(f"\n🔍 Model: {model_name}")
    print(f"   → fit():     {'YES' if has_fit else 'NO'}")
    print(f"   → predict(): {'YES' if has_predict else 'NO'}")

    if not (has_fit and has_predict):
        raise TypeError(
            f"❌ Model '{model_name}' does NOT implement sklearn API "
            f"(fit={has_fit}, predict={has_predict})"
        )

    # ------------------------------------------------------------
    # 2️⃣ TRAIN
    # ------------------------------------------------------------
    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 3️⃣ Task-specific prediction & metrics
    # ------------------------------------------------------------
    evaluator = get_task_evaluator(task)

    y_pred, extra_pred_outputs = evaluator.predict(model, X_test)
    metrics = evaluator.evaluate(y_test, y_pred, extra_pred_outputs)

    # ------------------------------------------------------------
    # 4️⃣ PRINT COEFFICIENTS IF AVAILABLE
    # ------------------------------------------------------------
    print("📊 Metrics:")
    for k, v in metrics.items():
        print(f"   {k} = {v:.6f}" if isinstance(v, float) else f"   {k} = {v}")

    # ------------------------------------------------------------
    # 5️⃣ PRINT COEFFICIENTS IF AVAILABLE
    # ------------------------------------------------------------
    has_coef = hasattr(model, "coef_")
    has_importance = hasattr(model, "feature_importances_")

    if has_coef:
        print("\n🧠 Learned Coefficients:")
        print(model.coef_)

    elif has_importance:
        print("\n🌲 Feature Importances:")
        print(model.feature_importances_)

    else:
        print("\n⚠️ Model does not expose coefficients or feature_importances_")

    # ------------------------------------------------------------
    # 6️⃣ LOG TO MLFLOW
    # ------------------------------------------------------------
    log_with_mlflow(
        run_name=model_name,
        model=model,
        params=model_params or {},
        metrics=metrics,
        pre=pre,
        experiment=mlflow_experiment,
        col_manager=col_manager,
        model_mlflow_cfg=model_mlflow_cfg,
    )

    return {
        "metrics": metrics
    }
