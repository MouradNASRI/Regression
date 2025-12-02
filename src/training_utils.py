# src/training_utils.py

from typing import Dict, Any
from sklearn.metrics import mean_squared_error, r2_score
from src.model_registry import get_model
from src.mlflow_utils import log_with_mlflow


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
) -> Dict[str, Any]:
    """
    Generic training routine for sklearn-compatible regression models.
    Performs:
      - model construction (via registry)
      - fit() on train data
      - predict() on test data
      - MSE + R² evaluation
      - MLflow logging (with optional preprocessor pipeline)

    Returns:
      (mse, r2)
    """

    # ------------------------------------------------------------
    # 1️⃣ BUILD MODEL
    # ------------------------------------------------------------
    model = get_model(model_name, **(model_params or {}))

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
    # 3️⃣ EVALUATE
    # ------------------------------------------------------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"📊 Results →  MSE={mse:.6f}   R²={r2:.6f}")
    # ------------------------------------------------------------
    # 3️⃣ PRINT COEFFICIENTS IF AVAILABLE
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
    # 4️⃣ LOG TO MLFLOW
    # ------------------------------------------------------------
    log_with_mlflow(
        run_name=model_name,
        model=model,
        params=model_params or {},
        metrics={"mse": float(mse), "r2": float(r2)},
        pre=pre,
        experiment=mlflow_experiment,
        col_manager=col_manager,
        model_mlflow_cfg=model_mlflow_cfg,
    )

    return {
        "metrics": {
            "mse": mse,
            "r2": r2,
        }
    }
