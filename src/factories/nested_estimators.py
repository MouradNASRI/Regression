from __future__ import annotations
from typing import Any, Dict
from sklearn.base import BaseEstimator

from sklearn.tree import DecisionTreeClassifier


def build_nested_estimator(spec: Dict[str, Any]) -> BaseEstimator:
    """
    Build a nested estimator from a YAML spec like:
      {"type": "decision_tree", "params": {...}}

    Extend this mapping over time as you add more nested estimators.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"Invalid nested estimator spec: {spec}")

    est_type = str(spec["type"]).lower()
    est_params = spec.get("params", {}) or {}

    if est_type == "decision_tree":
        return DecisionTreeClassifier(**est_params)

    raise ValueError(f"Unknown nested estimator type '{est_type}' in spec: {spec}")
