from typing import Optional

from src.model_registry import regression as reg
from src.model_registry import classification as clf

_TASK_REGISTRIES = {
    "regression": reg,
    "classification": clf,
}

def get_registry(task: str):
    task = task.lower()
    if task not in _TASK_REGISTRIES:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_TASK_REGISTRIES)}")
    return _TASK_REGISTRIES[task]

def get_model(task: str, name: str, **kwargs):
    return get_registry(task).get_model(name, **kwargs)

def list_models(task: str, by_group: bool = True):
    return get_registry(task).list_models(by_group=by_group)

def get_model_group(task: str, name: str) -> Optional[str]:
    return get_registry(task).get_model_group(name)
