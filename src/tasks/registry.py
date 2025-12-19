from typing import Dict

from .base import BaseTaskEvaluator
from .regression import RegressionTaskEvaluator
from .classification import ClassificationTaskEvaluator


# Global registry: maps task name → evaluator instance
TASK_EVALUATORS: Dict[str, BaseTaskEvaluator] = {
    "regression": RegressionTaskEvaluator(),
    "classification": ClassificationTaskEvaluator(),
    # Later:
    # "ranking": RankingTaskEvaluator(),
    # "clustering": ClusteringTaskEvaluator(),
    # "poisson_regression": PoissonTaskEvaluator(),
    # etc.
}


def get_task_evaluator(task: str) -> BaseTaskEvaluator:
    """
    Retrieve the evaluator for the given task.

    Raises a clear error if the task is unknown.
    """
    task = task.lower()
    if task not in TASK_EVALUATORS:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Available: {list(TASK_EVALUATORS.keys())}"
        )
    return TASK_EVALUATORS[task]
