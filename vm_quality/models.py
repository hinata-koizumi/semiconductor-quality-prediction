
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

def build_ridge_pipeline(alpha: float) -> Pipeline:
    """
    Returns the standard pipeline: StandardScaler -> Ridge.
    Note: Feature construction (Poly) is handled in features.py manually.
    """
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    )
