from typing import Optional
from src.models.regression.linear import LINEAR_REGISTRY
from src.models.regression.trees import TREE_REGISTRY
from src.models.regression.kernel import KERNEL_REGISTRY
from src.models.regression.instance import INSTANCE_REGISTRY
from src.models.regression.nn import NN_REGISTRY
from src.models.regression.robust import ROBUST_REGISTRY
from src.models.regression.glm import GLM_REGISTRY
from src.models.regression.special import SPECIAL_REGISTRY


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

def get_model(name: str, **kwargs):
    """
    Pattern A: model construction is handled by src.factories.spec_builder.build_from_spec.
    This function is kept only for validation / backward compatibility.
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown regression model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    # Return the class (not an instance). kwargs ignored.
    return MODEL_REGISTRY[key]


def list_models(by_group: bool = True):
    return GROUPS if by_group else sorted(MODEL_REGISTRY.keys())

def get_model_group(name: str) -> Optional[str]:
    key = name.lower()
    for group_name, registry in GROUPS.items():
        if key in registry:
            return group_name
    return None
