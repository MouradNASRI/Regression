from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor

GLM_REGISTRY = {
    "poisson": PoissonRegressor,  # count data, positive targets
    "gamma": GammaRegressor,      # continuous positive targets
    "tweedie": TweedieRegressor,  # flexible GLM family
}
