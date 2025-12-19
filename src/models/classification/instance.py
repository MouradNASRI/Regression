"""
Classification - Instance-based models registry.
"""

from sklearn.neighbors import KNeighborsClassifier

INSTANCE_REGISTRY = {
    "knn_clf": KNeighborsClassifier,
}
