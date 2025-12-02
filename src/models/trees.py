from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor
)

TREE_REGISTRY = {
    "tree": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "extratrees": ExtraTreesRegressor,
    "gboost": GradientBoostingRegressor,
    "hgb": HistGradientBoostingRegressor,
    "ada": AdaBoostRegressor,
}
