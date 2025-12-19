from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

ROBUST_REGISTRY = {
    "huber": HuberRegressor,
    "ransac": RANSACRegressor,
    "theilsen": TheilSenRegressor,
}
