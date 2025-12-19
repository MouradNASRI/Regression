from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    SGDRegressor,
)

LINEAR_REGISTRY = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "lars": Lars,
    "lassolars": LassoLars,
    "omp": OrthogonalMatchingPursuit,
    "sgd": SGDRegressor,
}
