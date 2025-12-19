from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseTaskEvaluator(ABC):
    """
    Abstract base class for task-specific evaluation logic.

    Each concrete task (regression, classification, ...) implements:
      - predict(...)
      - evaluate(...)
    """

    @abstractmethod
    def predict(self, model, X_test) -> Tuple[Any, Dict[str, Any]]:
        """
        Run the appropriate prediction method(s) for this task.

        Returns
        -------
        y_pred : array-like
            Main predictions (e.g. regression outputs or class labels).
        extra : dict
            Extra outputs (e.g. probabilities for classification).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, y_true, y_pred, extra: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute metrics for this task and return them as a dict.
        """
        raise NotImplementedError
