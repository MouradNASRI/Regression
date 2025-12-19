"""
Classification model registry.

This module defines:
- group-level registries (linear, trees, kernel, etc.)
- a unified MODEL_REGISTRY for classification
- helper functions to:
    * instantiate a model
    * list available models
    * retrieve the group of a model

This registry is selected at runtime based on the `--task classification` CLI flag.
"""

from typing import Optional

# ------------------------------------------------------------------
# Import group-level classification registries
# Each registry maps a short model name -> sklearn estimator class
# ------------------------------------------------------------------

from src.models.classification.linear import LINEAR_REGISTRY
from src.models.classification.trees import TREES_REGISTRY
from src.models.classification.kernel import KERNEL_REGISTRY
from src.models.classification.instance import INSTANCE_REGISTRY
from src.models.classification.nn import NN_REGISTRY
from src.models.classification.robust import ROBUST_REGISTRY
from src.models.classification.special import SPECIAL_REGISTRY

# ------------------------------------------------------------------
# Group mapping
# ------------------------------------------------------------------

GROUPS = {
    "linear": LINEAR_REGISTRY,
    "trees": TREES_REGISTRY,
    "kernel": KERNEL_REGISTRY,
    "instance": INSTANCE_REGISTRY,
    "nn": NN_REGISTRY,
    "robust": ROBUST_REGISTRY,
    "special": SPECIAL_REGISTRY,
}

# ------------------------------------------------------------------
# Flattened registry: model_name -> estimator class
# ------------------------------------------------------------------

MODEL_REGISTRY = {
    model_name: estimator
    for group_registry in GROUPS.values()
    for model_name, estimator in group_registry.items()
}

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_model(name: str, **kwargs):
    """
    Instantiate a classification model by short name.

    Parameters
    ----------
    name : str
        Short model name (e.g. "logreg", "rf", "svc").
    **kwargs :
        Parameters forwarded to the sklearn estimator constructor.

    Returns
    -------
    model : sklearn estimator
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown classification model '{name}'. "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def list_models(by_group: bool = True):
    """
    List available classification models.

    Parameters
    ----------
    by_group : bool, default=True
        If True, return dict[group -> registry].
        If False, return sorted list of model names.

    Returns
    -------
    dict or list
    """
    return GROUPS if by_group else sorted(MODEL_REGISTRY.keys())


def get_model_group(name: str) -> Optional[str]:
    """
    Return the group name ("linear", "trees", "kernel", etc.)
    for a given classification model short name.

    Parameters
    ----------
    name : str
        Short model name.

    Returns
    -------
    group_name : str or None
        Group name if found, else None.
    """
    key = name.lower()
    for group_name, registry in GROUPS.items():
        if key in registry:
            return group_name
    return None
