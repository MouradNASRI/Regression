from sklearn.linear_model import OrthogonalMatchingPursuit, Lars, LassoLars
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression

SPECIAL_REGISTRY = {
    "omp": OrthogonalMatchingPursuit,
    "lars": Lars,
    "lassolars": LassoLars,
    "pls": PLSRegression,
    "isotonic": IsotonicRegression,
}
