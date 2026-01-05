"""
Spec Builder

This module converts YOUR YAML spec (type/params) into a real sklearn estimator.

Why this exists:
- You want a scalable way to support nested estimators (AdaBoost.estimator, etc.)
- You want sklearn routing params like 'estimator__max_depth' to work
- You want to avoid if/else ladders by using a central catalog + dynamic imports

Spec format (examples)

Simple:
  {"type": "logreg", "params": {"C": 1.0, "max_iter": 1000}}

Nested estimator:
  {"type": "ada_clf",
   "params": {
       "n_estimators": 200,
       "estimator": {"type": "tree_clf", "params": {"max_depth": 2}},
       "estimator__max_depth": 3
   }
  }

Rules:
- Any dict value that contains "type" is treated as a nested estimator spec
  and is built recursively.
- Any param key containing '__' is applied via set_params after instantiation.
"""

from __future__ import annotations
from typing import Any, Dict

from sklearn.base import BaseEstimator

from src.factories.estimator_catalog import ESTIMATOR_CATALOG
from src.factories.import_utils import import_class
from src.factories.sklearn_factory import split_init_and_post_params


def build_from_spec(spec: Dict[str, Any]) -> BaseEstimator:
    """
    Build an estimator from a spec dict:
      spec = {"type": <catalog_key>, "params": {...}}

    Returns:
      A fully constructed sklearn estimator with post-init params applied.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"Invalid estimator spec (expected dict with 'type'): {spec}")

    est_type = str(spec["type"]).lower().strip()
    params = spec.get("params", {}) or {}

    if est_type not in ESTIMATOR_CATALOG:
        # Fail fast with a helpful message (prevents silent misconfig)
        known = ", ".join(sorted(ESTIMATOR_CATALOG.keys()))
        raise KeyError(
            f"Unknown estimator type '{est_type}'. "
            f"Add it to ESTIMATOR_CATALOG. Known types: {known}"
        )

    # 1) Resolve class from catalog
    cls_path = ESTIMATOR_CATALOG[est_type]
    cls = import_class(cls_path)

    # 2) Build any nested estimator specs inside params (recursively)
    params = materialize_nested_specs(params)

    # 3) Split init kwargs vs post-init kwargs (sklearn routing)
    init_kwargs, post_kwargs = split_init_and_post_params(params)

    # 4) Instantiate
    est: BaseEstimator = cls(**init_kwargs)

    # 5) Apply nested routing params (e.g. estimator__max_depth)
    if post_kwargs:
        est.set_params(**post_kwargs)

    return est


def materialize_nested_specs(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk a dict of params and replace any "spec-like" values with real estimators.

    A "spec-like" value is a dict containing:
      - "type": <catalog_key>
      - optional "params": {...}

    This keeps the system generic: it doesn't care whether the nested estimator
    is called 'estimator', 'final_estimator', 'base_estimator', etc.
    If the value looks like a spec, we build it.
    """
    out: Dict[str, Any] = {}

    for k, v in (params or {}).items():
        if isinstance(v, dict) and "type" in v:
            out[k] = build_from_spec(v)  # recursion
        else:
            out[k] = v

    return out
