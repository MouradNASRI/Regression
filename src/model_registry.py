from typing import Optional

from src.models.linear import LINEAR_REGISTRY
from src.models.trees import TREE_REGISTRY
from src.models.kernel import KERNEL_REGISTRY
from src.models.instance import INSTANCE_REGISTRY
from src.models.nn import NN_REGISTRY
from src.models.robust import ROBUST_REGISTRY
from src.models.glm import GLM_REGISTRY
from src.models.special import SPECIAL_REGISTRY

GROUPS = {
    "linear": LINEAR_REGISTRY,
    "trees": TREE_REGISTRY,
    "kernel": KERNEL_REGISTRY,
    "instance": INSTANCE_REGISTRY,
    "nn": NN_REGISTRY,
    "robust": ROBUST_REGISTRY,
    "glm": GLM_REGISTRY,
    "special": SPECIAL_REGISTRY,
}

MODEL_REGISTRY = {k: v for g in GROUPS.values() for k, v in g.items()}

def get_model(name, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)

def list_models(by_group=True):
    return GROUPS if by_group else sorted(MODEL_REGISTRY.keys())

def get_model_group(name: str) -> Optional[str]:
    """
    Return the group name ('linear', 'trees', 'kernel', etc.)
    for a given model short name, or None if not found.
    """
    key = name.lower()
    for group_name, registry in GROUPS.items():
        if key in registry:
            return group_name
    return None