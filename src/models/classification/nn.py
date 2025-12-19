"""
Classification - Neural-network models registry.
"""

from sklearn.neural_network import MLPClassifier

NN_REGISTRY = {
    "mlp_clf": MLPClassifier,
}
