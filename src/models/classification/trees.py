"""
Classification - Tree-based models registry.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)

TREES_REGISTRY = {
    "tree_clf": DecisionTreeClassifier,
    "rf_clf": RandomForestClassifier,
    "extratrees_clf": ExtraTreesClassifier,
    "gboost_clf": GradientBoostingClassifier,
    "ada_clf": AdaBoostClassifier,
    "hgb_clf": HistGradientBoostingClassifier,
}
