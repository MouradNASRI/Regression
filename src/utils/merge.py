"""
Configuration merge utilities.

This module defines merge helpers used when combining:
- model YAML defaults
- hyperparameter search defaults
- CLI overrides
- per-trial parameter updates

Design principles
-----------------
- Dicts are merged recursively.
- Non-dict values overwrite earlier values.
- The right-hand side always wins.
- No mutation of inputs.
"""

from __future__ import annotations
from typing import Any, Dict


def deep_merge(base: Dict[str, Any] | None,
               override: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Rules:
    - If both base[k] and override[k] are dicts → recurse.
    - Otherwise override[k] replaces base[k].
    - None is treated as an empty dict.
    - Inputs are NOT mutated.

    Examples
    --------
    >>> deep_merge(
    ...   {"a": 1, "b": {"x": 10, "y": 20}},
    ...   {"b": {"y": 99}, "c": 3}
    ... )
    {'a': 1, 'b': {'x': 10, 'y': 99}, 'c': 3}

    >>> deep_merge(
    ...   {"estimator": {"type": "tree_clf", "params": {"max_depth": 1}}},
    ...   {"estimator__max_depth": 3}
    ... )
    {'estimator': {'type': 'tree_clf', 'params': {'max_depth': 1}},
     'estimator__max_depth': 3}
    """
    if base is None:
        base = {}
    if override is None:
        override = {}

    result: Dict[str, Any] = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result
