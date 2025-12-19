from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .base import BaseTaskEvaluator


class ClassificationTaskEvaluator(BaseTaskEvaluator):
    """
    Evaluation logic for (binary or multiclass) classification tasks.
    """

    def predict(self, model, X_test) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        For classification, we may want both class labels and probabilities.
        """
        y_pred = model.predict(X_test)

        extra: Dict[str, Any] = {}

        # If probabilities are available, store them for metrics like ROC-AUC.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            extra["proba"] = proba

        return np.asarray(y_pred), extra

    def evaluate(self, y_true, y_pred, extra: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute classification metrics.

        This implementation uses:
          - accuracy
          - weighted F1
          - ROC-AUC for binary classification (if probabilities available)
        """
        metrics: Dict[str, float] = {}

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        metrics["accuracy"] = float(acc)
        metrics["f1"] = float(f1)

        # Optional ROC-AUC for binary classification if probabilities exist.
        proba = extra.get("proba")
        if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
            metrics["roc_auc"] = float(auc)

        return metrics
