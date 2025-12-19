"""
Classification - Robust / linear-ish models registry.

There is no direct "RANSACClassifier" or "TheilSenClassifier" in sklearn.
Here we include robust-ish linear classifiers that behave well with noise/outliers.
"""

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

ROBUST_REGISTRY = {
    # Huber loss classification isn't provided as "HuberClassifier" in sklearn,
    # but SGDClassifier supports robust-ish losses and regularization.
    "sgd_robust_clf": SGDClassifier,

    # LinearSVC is often used as a robust linear baseline.
    "linearsvc_robust": LinearSVC,
}
