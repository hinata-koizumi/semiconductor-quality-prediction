
import pandas as pd
import numpy as np
from typing import List, Tuple

def build_matrix(df_slice: pd.DataFrame, features: List[str], t_mode: str = "abs", ref_time: pd.Timestamp = None, interactions: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs the design matrix [t, t^2, X...].
    If interactions=True, adds [t*X] terms.
    Returns (matrix, column_names).
    """
    X_raw = df_slice[features].values
    
    # Time Feature
    if t_mode == "abs":
        # Absolute Time scaling (Days since 2000-01-01) - Robust Baseline
        t_vals = (df_slice["process_end_time"] - pd.Timestamp("2000-01-01")).dt.total_seconds() / 86400.0
    elif t_mode == "rel":
        # Relative to ref_time
        if ref_time is None:
            raise ValueError("ref_time is required for t_mode='rel'")
        t_vals = (df_slice["process_end_time"] - ref_time).dt.total_seconds() / 86400.0
    else:
        raise ValueError(f"Unknown t_mode: {t_mode}")

    t_vals = t_vals.values.reshape(-1, 1)
    t_sq = t_vals ** 2
    
    # Stack: [t, t^2, X]
    if interactions:
        # t * X for each feature
        t_X = X_raw * t_vals
        X_out = np.hstack([t_vals, t_sq, X_raw, t_X])
        col_names = ["t", "t^2"] + features + [f"t_{f}" for f in features]
    else:
        X_out = np.hstack([t_vals, t_sq, X_raw])
        col_names = ["t", "t^2"] + features
    
    # Verify dimensions
    assert X_out.shape[1] == len(col_names)
    
    return X_out, col_names
