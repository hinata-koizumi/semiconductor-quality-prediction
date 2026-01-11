
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Add project root to path (not needed if running from root)
# sys.path.append(str(Path(__file__).parent.parent))

from vm_quality.data import load_dataset, SplitSpec
from vm_quality.eval import run_cv, run_final_test, run_ablation, run_experiment_suite, ModelConfig
from vm_quality.viz import plot_ov_timeseries, plot_rmse_vs_train_size, plot_pred_vs_true, plot_residuals

def main():
    # Setup Artifacts Dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(f"artifacts/run_{ts}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating Artifacts in: {artifact_dir}")
    
    # Config
    data_path = Path("data/kadai.xlsx")
    df = load_dataset(data_path)
    spec = SplitSpec(test_start=1776, test_size=500)
    features = ['X32', 'X36', 'X27', 'X83']
    all_features = [f"X{i}" for i in range(1, 84)]
    
    # Final Config
    base_config = ModelConfig(name="Final", t_mode="abs", alpha=500.0)
    
    # 1. Run CV
    print("Running Rolling CV...")
    cv_res = run_cv(df, spec, features, base_config)
    pd.DataFrame({"fold": range(1, 6), "rmse": cv_res["fold_rmses"]}).to_csv(artifact_dir / "cv_folds.csv", index=False)
    # Save fold details
    pd.DataFrame(cv_res["fold_details"]).to_csv(artifact_dir / "cv_fold_details.csv", index=False)
    
    fold_details = sorted(cv_res["fold_details"], key=lambda d: d["fold"])
    train_sizes = [detail["train_size"] for detail in fold_details]
    plot_rmse_vs_train_size(cv_res["fold_rmses"], train_sizes, artifact_dir / "cv_rmse_scatter.png")
    
    # 2. Run Final Test & DataGuard
    print("Running Final Test...")
    test_res = run_final_test(df, spec, features, base_config)

    # 2b. Baselines (fixed split, same cutoff)
    print("Running Baseline-0 (RF sample)...")
    final_test = df.iloc[spec.test_start : spec.test_start + spec.test_size].copy()
    cutoff_time = final_test["process_end_time"].iloc[0]
    full_train = df.iloc[: spec.test_start].copy()
    full_train = full_train[full_train["final_mes_time"] < cutoff_time].copy()

    X_train = full_train[all_features].values
    y_train = full_train["OV"].values
    X_test = final_test[all_features].values
    y_test = final_test["OV"].values

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    pred_rf = np.maximum(pred_rf, 0)
    baseline0_rmse = np.sqrt(mean_squared_error(y_test, pred_rf))

    print("Running Baseline-1 (All vars Ridge)...")
    baseline1_config = ModelConfig(name="Baseline-1", t_mode="abs", alpha=500.0)
    baseline1_res = run_final_test(df, spec, all_features, baseline1_config)
    baseline1_rmse = baseline1_res["test_rmse"]
    
    # Save Coefficients
    coef_df = pd.DataFrame(list(test_res["coefficients"].items()), columns=["Feature", "Coefficient"])
    # Sort by absolute value
    coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoef", ascending=False)
    coef_df.to_csv(artifact_dir / "coefficients.csv", index=False)
    
    # Save Metrics
    final_rmse = test_res["test_rmse"]
    improvement_vs_baseline1 = None
    if baseline1_rmse != 0:
        improvement_vs_baseline1 = 1.0 - (final_rmse / baseline1_rmse)

    metrics = {
        "cv_mean": cv_res["mean_rmse"],
        "cv_std": cv_res["std_rmse"],
        "cv_last_fold": cv_res["last_fold_rmse"],
        "test_rmse": final_rmse,
        "dataguard_margin": str(test_res["guard"]["margin"]),
        "features": features,
        "baseline0_rmse": baseline0_rmse,
        "baseline1_rmse": baseline1_rmse,
        "final_rmse": final_rmse,
        "improvement_rate_vs_baseline1": improvement_vs_baseline1
    }
    pd.Series(metrics).to_csv(artifact_dir / "metrics.csv")
    
    # Plotting
    plot_ov_timeseries(
        test_res["process_end_times"],
        test_res["y_true"],
        test_res["predictions"],
        artifact_dir / "ov_timeseries.png",
    )
    plot_pred_vs_true(test_res["y_true"], test_res["predictions"], artifact_dir / "pred_vs_true.png")
    plot_residuals(test_res["y_true"], test_res["predictions"], test_res["process_end_times"], artifact_dir / "residuals_time.png")
    
    # 3. Ablation
    print("Running Ablation Study...")
    abl_df = run_ablation(df, spec, features, base_config)
    abl_df.to_csv(artifact_dir / "ablation.csv", index=False)
    
    # 4. Drift Experiments (Real execution)
    print("Running Drift Experiment Suite...")
    drift_df = run_experiment_suite(df, spec, features)
    drift_df.to_csv(artifact_dir / "experiments.csv", index=False)
    
    print("Done! All artifacts generated.")

if __name__ == "__main__":
    main()
