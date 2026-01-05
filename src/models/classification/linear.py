"""
Classification - Linear models registry.

Short names are chosen to avoid collisions with regression names.
"""

from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)

from sklearn.svm import LinearSVC 

LINEAR_REGISTRY = {
    "logreg": LogisticRegression,
    "sgd_clf": SGDClassifier,
    "ridge_clf": RidgeClassifier,
    "pa_clf": PassiveAggressiveClassifier,
    "perceptron": Perceptron,
    "linearsvc": LinearSVC,
}
