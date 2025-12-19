from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .base import BaseTaskEvaluator


class RegressionTaskEvaluator(BaseTaskEvaluator):
    """
    Evaluation logic for standard regression tasks.
    """

    def predict(self, model, X_test) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        For regression, we typically just need model.predict().
        """
        y_pred = model.predict(X_test)
        return np.asarray(y_pred), {}

    def evaluate(self, y_true, y_pred, extra: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute common regression metrics (MSE, R²).

        You can extend this later with MAE, RMSE, etc.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": float(mse),
            "r2": float(r2),
        }
