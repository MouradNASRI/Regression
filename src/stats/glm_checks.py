import numpy as np
import pandas as pd
from scipy.stats import skew

def check_gamma_glm_suitability(y: pd.Series, verbose: bool = True) -> dict:
    """
    Diagnose whether Gamma regression is a suitable choice for the given target vector.

    Conditions for Gamma GLM:
      - All target values strictly > 0
      - Target distribution is right-skewed
      - Mean/variance relationship: var(y) approx proportional to mean(y)^2

    Returns:
      dict with suitability boolean and diagnostics
    """

    y_np = y.to_numpy().ravel()

    ######### 1. Strict Positivity #########
    strictly_positive = np.all(y_np > 0)

    ######### 2. Right Skewness #########
    skew_value = float(skew(y_np))
    right_skewed = skew_value > 1.0

    ######### 3. Mean–Variance Relationship #########
    mean_y = float(np.mean(y_np))
    var_y = float(np.var(y_np))
    # Gamma variance roughly ~ mean^2 (within tolerance)
    ratio = var_y / (mean_y ** 2) if mean_y > 0 else np.inf
    mv_relationship = 0.5 < ratio < 5.0  # loose threshold

    suitability = strictly_positive and right_skewed and mv_relationship

    diagnostics = {
        "strictly_positive": strictly_positive,
        "skewness": skew_value,
        "right_skewed": right_skewed,
        "mean": mean_y,
        "variance": var_y,
        "variance_to_mean2_ratio": ratio,
        "mean_variance_relationship_ok": mv_relationship,
        "gamma_glm_recommended": suitability,
    }

    if verbose:
        print("\n🔍 Gamma GLM Suitability Check")
        for k, v in diagnostics.items():
            print(f"  {k:35} = {v}")

    return diagnostics
