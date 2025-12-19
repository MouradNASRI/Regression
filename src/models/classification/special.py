"""
Classification - Special models registry.

These don't always fit neatly into the other categories.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

SPECIAL_REGISTRY = {
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "gnb": GaussianNB,

    # Calibration wrapper (useful when base estimator doesn't output well-calibrated probabilities)
    "calibrated_clf": CalibratedClassifierCV,
}
