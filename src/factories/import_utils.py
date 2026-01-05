"""
Import utilities.

We keep dynamic imports in one place so the rest of the code stays clean.
"""

import importlib
from typing import Type


def import_class(dotted_path: str) -> Type:
    """
    Import a class from a dotted path.

    Example:
      import_class("sklearn.ensemble.AdaBoostClassifier")
        -> <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>
    """
    module_path, cls_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
