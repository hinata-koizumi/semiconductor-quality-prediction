
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_ov_timeseries(time_axis, y_true, y_pred, output_path: Path):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_top, ax_bottom = axes

    x_vals = np.arange(len(y_true))
    y_true_line = y_true.astype(float).copy()

    if len(time_axis) > 1:
        t_vals = pd.to_datetime(pd.Series(time_axis), errors="coerce")
        diffs_all = t_vals.diff().dt.total_seconds().values
        diffs = diffs_all[np.isfinite(diffs_all)]
        if diffs.size > 0:
            median_diff = np.median(diffs)
            if median_diff > 0:
                gap_threshold = median_diff * 3.0
                gap_indices = np.where(diffs_all > gap_threshold)[0]
                for idx in gap_indices:
                    if 0 <= idx < len(y_true_line):
                        y_true_line[idx] = np.nan

    ax_top.plot(x_vals, y_true_line, color="#1f77b4", linewidth=1.1, alpha=0.5)
    ax_top.scatter(x_vals, y_true, label="True (OV)", color="#1f77b4", s=14, alpha=0.85)
    ax_top.scatter(x_vals, y_pred, label="Pred (OV)", color="#ff7f0e", s=10, alpha=0.5)

    ax_top.set_ylabel("OV")
    ax_top.set_title("OV Time Series in Evaluation Window")
    ax_top.legend(frameon=True)
    ax_top.grid(True, alpha=0.2)

    resid_mean = float(np.nanmean(residuals))
    resid_median = float(np.nanmedian(residuals))
    ax_bottom.scatter(x_vals, residuals, color="#2ca02c", s=12, alpha=0.8)
    ax_bottom.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_bottom.axhline(resid_mean, color="gray", linestyle=":", linewidth=1)
    ax_bottom.axhline(resid_median, color="gray", linestyle="-.", linewidth=1)
    ax_bottom.text(
        0.99,
        0.95,
        f"mean={resid_mean:.1f}\nmedian={resid_median:.1f}",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="gray",
        bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
    )
    ax_bottom.set_xlabel(f"Index (0-{len(y_true) - 1})")
    ax_bottom.set_ylabel("Residual")
    ax_bottom.grid(True, alpha=0.2)

    if len(x_vals) > 0:
        ax_bottom.set_xlim(0, len(x_vals) - 1)
        step = 100 if len(x_vals) > 100 else max(len(x_vals) // 5, 1)
        ticks = list(range(0, len(x_vals), step))
        if (len(x_vals) - 1) not in ticks:
            ticks.append(len(x_vals) - 1)
        ax_bottom.set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_rmse_by_fold(fold_rmses: list, output_path: Path, train_sizes: list = None):
    plt.figure(figsize=(8, 5))
    folds = list(range(1, len(fold_rmses) + 1))
    plt.bar(folds, fold_rmses, color="skyblue")
    plt.axhline(np.mean(fold_rmses), color="orange", linestyle="--", label="Mean RMSE")
    if train_sizes is not None and len(train_sizes) == len(fold_rmses):
        labels = [f"F{fold}\n(n={size})" for fold, size in zip(folds, train_sizes)]
        plt.xticks(folds, labels)
        plt.xlabel("Fold (train samples)")
    else:
        plt.xticks(folds)
        plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE by Fold (Drift Check)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rmse_vs_train_size(fold_rmses: list, train_sizes: list, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.scatter(train_sizes, fold_rmses, color="skyblue")
    for i, (size, rmse) in enumerate(zip(train_sizes, fold_rmses), start=1):
        plt.annotate(f"F{i}", (size, rmse), textcoords="offset points", xytext=(4, 4), fontsize=9)
    plt.xlabel("Train Samples")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Train Samples (Rolling CV)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pred_vs_true(y_true, y_pred, output_path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel("True OV")
    plt.ylabel("Predicted OV")
    plt.title("Test: Predicted vs True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_residuals(y_true, y_pred, time_axis, output_path: Path):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(time_axis, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Residuals over Time")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
