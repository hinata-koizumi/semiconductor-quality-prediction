
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from .data import SplitSpec, apply_dataguard
from .features import build_matrix
from .models import build_ridge_pipeline

@dataclass
class ModelConfig:
    name: str
    t_mode: str = "abs"
    interactions: bool = False
    weight_k: float = 0.0
    log_target: bool = False
    alpha: float = 500.0

@dataclass
class EvalMetrics:
    cv_mean: float
    cv_std: float
    cv_last_fold: float
    test_rmse: float
    fold_rmses: List[float]

def calculate_weights(df_slice: pd.DataFrame, ref_time: pd.Timestamp, k: float) -> np.ndarray:
    if k <= 0:
        return None
    # time delta in days (negative values)
    dt_days = (df_slice["process_end_time"] - ref_time).dt.total_seconds() / 86400.0
    # exp(k * t_rel) where t_rel is negative or 0
    weights = np.exp(k * dt_days)
    return weights.values

def run_cv(df: pd.DataFrame, spec: SplitSpec, features: List[str], config: ModelConfig, n_splits: int = 5) -> Dict:
    full_train_candidates = df.iloc[: spec.test_start].copy()
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=150)
    fold_rmses = []
    
    fold_details = []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(full_train_candidates)):
        tr_fold = full_train_candidates.iloc[train_idx]
        val_fold = full_train_candidates.iloc[val_idx]
        
        # Validation Start Point acts as the "Reference Time" for this fold
        val_start_time = val_fold["process_end_time"].iloc[0]
        
        # Strict Filtering
        mask = tr_fold["final_mes_time"] < val_start_time
        tr_filtered = tr_fold[mask]
        
        fold_details.append({
            "fold": fold_i+1,
            "train_size": len(tr_filtered),
            "val_size": len(val_fold),
            "val_start": val_start_time
        })
        
        # Features
        X_tr, _ = build_matrix(tr_filtered, features, t_mode=config.t_mode, ref_time=val_start_time, interactions=config.interactions)
        X_val, _ = build_matrix(val_fold, features, t_mode=config.t_mode, ref_time=val_start_time, interactions=config.interactions)
        
        y_tr = tr_filtered["OV"].values
        y_val = val_fold["OV"].values
        
        if config.log_target:
            y_tr = np.log1p(y_tr)
            
        weights = calculate_weights(tr_filtered, val_start_time, config.weight_k)
        
        # Train
        model = build_ridge_pipeline(config.alpha)
        if weights is not None:
            model.fit(X_tr, y_tr, ridge__sample_weight=weights)
        else:
            model.fit(X_tr, y_tr)
        
        pred = model.predict(X_val)
        if config.log_target:
            pred = np.expm1(pred)
            
        pred = np.maximum(pred, 0)
        
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        fold_rmses.append(rmse)
        
    return {
        "mean_rmse": np.mean(fold_rmses),
        "std_rmse": np.std(fold_rmses),
        "last_fold_rmse": fold_rmses[-1],
        "fold_rmses": fold_rmses,
        "fold_details": fold_details
    }

def run_final_test(df: pd.DataFrame, spec: SplitSpec, features: List[str], config: ModelConfig) -> Dict:
    final_test = df.iloc[spec.test_start : spec.test_start + spec.test_size].copy()
    cutoff_time = final_test["process_end_time"].iloc[0]
    
    full_train = df.iloc[: spec.test_start].copy()
    full_train = full_train[full_train["final_mes_time"] < cutoff_time].copy()
    
    guard_info = apply_dataguard(full_train, cutoff_time)
    
    # Features
    X_tr, col_names = build_matrix(full_train, features, t_mode=config.t_mode, ref_time=cutoff_time, interactions=config.interactions)
    y_tr = full_train["OV"].values
    
    X_test, _ = build_matrix(final_test, features, t_mode=config.t_mode, ref_time=cutoff_time, interactions=config.interactions)
    y_test_true = final_test["OV"].values

    if config.log_target:
        y_tr = np.log1p(y_tr)

    weights = calculate_weights(full_train, cutoff_time, config.weight_k)

    model = build_ridge_pipeline(config.alpha)
    if weights is not None:
        model.fit(X_tr, y_tr, ridge__sample_weight=weights)
    else:
        model.fit(X_tr, y_tr)
    
    pred = model.predict(X_test)
    if config.log_target:
        pred = np.expm1(pred)

    pred = np.maximum(pred, 0)
    rmse = np.sqrt(mean_squared_error(y_test_true, pred))
    
    # Extract Coefficients (Standardized)
    # Pipeline: [scaler, ridge]
    ridge_model = model.named_steps['ridge']
    coefs = ridge_model.coef_
    
    return {
        "test_rmse": rmse,
        "predictions": pred,
        "y_true": y_test_true,
        "residuals": y_test_true - pred,
        "guard": guard_info,
        "process_end_times": final_test["process_end_time"].values,
        "coefficients": dict(zip(col_names, coefs))
    }

def run_ablation(df: pd.DataFrame, spec: SplitSpec, features: List[str], config: ModelConfig) -> pd.DataFrame:
    results = []
    
    # Full
    print("  Running Ablation: Full")
    cv_res = run_cv(df, spec, features, config)
    test_res = run_final_test(df, spec, features, config)
    results.append({
        "Condition": "Full", "Removed": "-", 
        "CV Mean": cv_res["mean_rmse"], "CV Last": cv_res["last_fold_rmse"], "Test RMSE": test_res["test_rmse"]
    })
    
    for f in features:
        print(f"  Running Ablation: Minus {f}")
        subset = [x for x in features if x != f]
        cv_res_sub = run_cv(df, spec, subset, config)
        test_res_sub = run_final_test(df, spec, subset, config)
        results.append({
            "Condition": f"Minus {f}", "Removed": f,
            "CV Mean": cv_res_sub["mean_rmse"], "CV Last": cv_res_sub["last_fold_rmse"], "Test RMSE": test_res_sub["test_rmse"]
        })
    return pd.DataFrame(results)

def run_experiment_suite(df: pd.DataFrame, spec: SplitSpec, features: List[str]) -> pd.DataFrame:
    """
    Runs the comprehensive suite of models for Table 8.
    """
    configs = [
        ModelConfig(name="Base (Abs)", t_mode="abs", alpha=500.0),
        ModelConfig(name="RelTime", t_mode="rel", alpha=500.0),
        ModelConfig(name="Decay Weight", t_mode="abs", weight_k=0.05, alpha=500.0), # picked 0.05 as representative
        ModelConfig(name="Log Target", t_mode="abs", log_target=True, alpha=500.0),
        ModelConfig(name="Interactions", t_mode="abs", interactions=True, alpha=500.0),
    ]
    
    results = []
    for cfg in configs:
        print(f"  Running Exp: {cfg.name}")
        cv = run_cv(df, spec, features, cfg)
        test = run_final_test(df, spec, features, cfg)
        results.append({
            "Model Config": cfg.name,
            "CV Last-Fold": cv["last_fold_rmse"],
            "Test RMSE": test["test_rmse"],
            "Conclusion": "TBD"
        })
        
    return pd.DataFrame(results)
