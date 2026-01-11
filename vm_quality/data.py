
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SplitSpec:
    test_start: int = 1776
    test_size: int = 500

def load_dataset(path: Path) -> pd.DataFrame:
    """Loads and sorts the dataset."""
    df = pd.read_excel(path)
    df["process_end_time"] = pd.to_datetime(df["process_end_time"])
    df["final_mes_time"] = pd.to_datetime(df["final_mes_time"])
    # strict monotonicity
    df = df.sort_values("process_end_time").reset_index(drop=True)
    return df

def get_cutoff_time(df: pd.DataFrame, spec: SplitSpec) -> pd.Timestamp:
    """Returns the process_end_time of the first test sample."""
    return df.iloc[spec.test_start]["process_end_time"]

def apply_dataguard(train_df: pd.DataFrame, cutoff_time: pd.Timestamp) -> dict:
    """
    Asserts requirement: ALL train measurements must occur BEFORE cutoff.
    Returns: dict with logging info.
    """
    if len(train_df) == 0:
        return {"result": "EMPTY", "margin": None}
        
    max_mes_time = train_df["final_mes_time"].max()
    margin = cutoff_time - max_mes_time
    
    assert max_mes_time < cutoff_time, \
        f"CRITICAL LEAKAGE: Found measure time {max_mes_time} >= cutoff {cutoff_time}"
    
    return {
        "cutoff_time": cutoff_time,
        "max_train_mes_time": max_mes_time,
        "margin": margin,
        "train_samples": len(train_df),
        "result": "PASSED"
    }
