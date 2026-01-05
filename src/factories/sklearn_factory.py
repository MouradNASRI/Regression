"""
sklearn utilities.

This module intentionally contains only sklearn-parameter mechanics.
It does NOT know anything about your spec schema (type/params),
and it does NOT import models.

The important convention:
- keys containing '__' (double underscore) are NOT constructor kwargs.
  They are meant for estimator.set_params(...) routing.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple


def split_init_and_post_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split params into:
      - init_kwargs: passable to ClassName(**init_kwargs)
      - post_kwargs: apply via estimator.set_params(**post_kwargs)

    Example:
      {"C": 1.0, "estimator__max_depth": 3}
        -> init_kwargs={"C": 1.0}
           post_kwargs={"estimator__max_depth": 3}
    """
    init_kwargs: Dict[str, Any] = {}
    post_kwargs: Dict[str, Any] = {}

    for k, v in (params or {}).items():
        if "__" in k:
            post_kwargs[k] = v
        else:
            init_kwargs[k] = v

    return init_kwargs, post_kwargs
