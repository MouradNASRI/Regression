"""
Classification - GLM-like models registry.

In sklearn, logistic regression is the main GLM classification estimator.
Other GLM families like Poisson/Gamma/Tweedie are for regression.

We keep this file for symmetry with regression structure and future extension.
"""

from sklearn.linear_model import LogisticRegression

GLM_REGISTRY = {
    "logreg_glm": LogisticRegression,
}
