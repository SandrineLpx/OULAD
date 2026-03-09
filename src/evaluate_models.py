"""
evaluate_models.py
==================
Post-training evaluation report with:
  - CV metric comparison chart
  - Calibration curve (reliability diagram)
  - Threshold analysis (precision / recall / F1 vs threshold)
  - Module-stratified performance (per-module recall + PR-AUC)
  - Demographic subgroup report (IMD band, age band, education level)
  - Permutation importance vs impurity importance comparison
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    recall_score,
)

from src.config import ARTIFACTS_DIR, MODELS_DIR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_artifact(name: str, default=None):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return default
    if name.endswith(".csv"):
        return pd.read_csv(path)
    if name.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_fig(path, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Core report
# ---------------------------------------------------------------------------

def create_evaluation_report() -> None:
    comparison_df = _load_artifact("model_comparison.csv", pd.DataFrame())
    holdout_df = _load_artifact("holdout_metrics.csv", pd.DataFrame())
    metadata = _load_artifact("best_model_metadata.json", {})
    threshold_report = _load_artifact("threshold_analysis.json", {})
    calibration_metrics = _load_artifact("calibration_metrics.json", {})
    temporal_metrics = _load_artifact("temporal_holdout_metrics.json", {})

    # ------------------------------------------------------------------
    # 1. CV metric comparison bar chart
    # ------------------------------------------------------------------
    if not comparison_df.empty:
        top5 = comparison_df[comparison_df["model"] != "baseline_dummy"] \
            .sort_values("cv_pr_auc_mean", ascending=False).head(5)
        plt.figure(figsize=(10, 5))
        x = np.arange(len(top5))
        w = 0.3
        plt.bar(x - w, top5["cv_f1_mean"], w, label="CV F1", color="#2980B9", alpha=0.85)
        plt.bar(x, top5["cv_pr_auc_mean"], w, label="CV PR-AUC", color="#E67E22", alpha=0.85)
        plt.bar(x + w, top5.get("cv_roc_auc_mean", pd.Series([0]*len(top5))), w,
                label="CV ROC-AUC", color="#27AE60", alpha=0.85)
        plt.xticks(x, top5["model"], rotation=25, ha="right")
        plt.ylabel("Score")
        plt.title("Top 5 Models: CV F1 / PR-AUC / ROC-AUC Comparison")
        plt.legend()
        _save_fig(ARTIFACTS_DIR / "top_model_metric_comparison.png")

    # ------------------------------------------------------------------
    # 2. Calibration curve
    # ------------------------------------------------------------------
    predictions_path = ARTIFACTS_DIR / "holdout_predictions.csv"
    y_test_path = ARTIFACTS_DIR / "y_test.csv"
    if predictions_path.exists() and y_test_path.exists():
        preds = pd.read_csv(predictions_path)
        y_test = pd.read_csv(y_test_path).squeeze()

        plt.figure(figsize=(7, 6))
        frac_pos, mean_pred = calibration_curve(y_test, preds["y_proba"], n_bins=10)
        plt.plot(mean_pred, frac_pos, "o-", color="#E74C3C", label="Model (uncalibrated)")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve (Reliability Diagram) — Holdout Set")
        plt.legend()
        _save_fig(ARTIFACTS_DIR / "reliability_diagram.png")

        # Threshold analysis
        prec, rec, thresholds = precision_recall_curve(y_test, preds["y_proba"])
        thr_ext = np.append(thresholds, 1.0)
        f1_curve = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        plt.figure(figsize=(8, 5))
        plt.plot(thr_ext, prec, label="Precision", color="#2E86AB")
        plt.plot(thr_ext, rec, label="Recall", color="#C0392B")
        plt.plot(thr_ext, f1_curve, label="F1", color="#27AE60")
        if threshold_report.get("threshold_max_f1"):
            plt.axvline(threshold_report["threshold_max_f1"], linestyle="--",
                        color="#27AE60", alpha=0.8, label=f"Opt F1 = {threshold_report['threshold_max_f1']:.2f}")
        if threshold_report.get("threshold_prec70_max_recall"):
            plt.axvline(threshold_report["threshold_prec70_max_recall"], linestyle=":",
                        color="#8E44AD", alpha=0.8,
                        label=f"Prec≥70% = {threshold_report['threshold_prec70_max_recall']:.2f}")
        plt.xlabel("Decision Threshold")
        plt.ylabel("Score")
        plt.title("Precision / Recall / F1 vs Decision Threshold")
        plt.legend()
        _save_fig(ARTIFACTS_DIR / "threshold_analysis_plot.png")

    # ------------------------------------------------------------------
    # 3. Module-stratified performance
    # ------------------------------------------------------------------
    _module_stratified_performance()

    # ------------------------------------------------------------------
    # 4. Demographic subgroup report
    # ------------------------------------------------------------------
    _demographic_subgroup_report()

    # ------------------------------------------------------------------
    # 5. Permutation importance vs impurity importance
    # ------------------------------------------------------------------
    _permutation_importance_comparison()

    # ------------------------------------------------------------------
    # 6. Temporal holdout summary
    # ------------------------------------------------------------------
    if temporal_metrics:
        _plot_temporal_summary(temporal_metrics, comparison_df)

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    report = {
        "best_model": metadata.get("best_model_name"),
        "best_metric": metadata.get("best_metric"),
        "best_metric_value": metadata.get("best_metric_value"),
        "threshold_max_f1": threshold_report.get("threshold_max_f1"),
        "threshold_prec70": threshold_report.get("threshold_prec70_max_recall"),
        "brier_before_calibration": calibration_metrics.get("brier_score_before_calibration"),
        "brier_after_calibration": calibration_metrics.get("brier_score_after_calibration"),
        "temporal_pr_auc_2014": temporal_metrics.get("pr_auc"),
        "top_cv_models": [] if comparison_df.empty else (
            comparison_df[comparison_df["model"] != "baseline_dummy"]
            .head(5).to_dict(orient="records")
        ),
        "top_holdout_models": [] if holdout_df.empty else (
            holdout_df.sort_values("pr_auc", ascending=False).head(5).to_dict(orient="records")
        ),
    }
    with open(ARTIFACTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Evaluation report generated.")


# ---------------------------------------------------------------------------
# Module-stratified performance
# ---------------------------------------------------------------------------

def _module_stratified_performance() -> None:
    pred_path = ARTIFACTS_DIR / "holdout_predictions.csv"
    x_path = ARTIFACTS_DIR / "X_test.csv"
    y_path = ARTIFACTS_DIR / "y_test.csv"
    if not (pred_path.exists() and x_path.exists() and y_path.exists()):
        return

    preds = pd.read_csv(pred_path)
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()

    if "code_module" not in X_test.columns:
        return

    rows = []
    threshold = 0.5
    for module, idx in X_test.groupby("code_module").groups.items():
        idx = list(idx)
        y_true_m = y_test.iloc[idx]
        y_proba_m = preds["y_proba"].iloc[idx]
        y_pred_m = (y_proba_m >= threshold).astype(int)
        n = len(y_true_m)
        if n < 20:
            continue
        rows.append({
            "code_module": module,
            "n_students": n,
            "withdrawal_rate": float(y_true_m.mean()),
            "recall": float(recall_score(y_true_m, y_pred_m, zero_division=0)),
            "f1": float(f1_score(y_true_m, y_pred_m, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true_m, y_proba_m))
            if y_true_m.nunique() > 1 else float("nan"),
        })

    if not rows:
        return

    module_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    module_df.to_csv(ARTIFACTS_DIR / "module_stratified_performance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(module_df))
    w = 0.35
    ax.bar(x - w / 2, module_df["recall"], w, label="Recall", color="#C0392B")
    ax.bar(x + w / 2, module_df["pr_auc"], w, label="PR-AUC", color="#2980B9")
    ax.set_xticks(x)
    ax.set_xticklabels(module_df["code_module"])
    ax.set_ylabel("Score")
    ax.set_title("Per-Module Performance: Recall and PR-AUC (threshold=0.5)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "module_stratified_performance.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Demographic subgroup report
# ---------------------------------------------------------------------------

def _demographic_subgroup_report() -> None:
    pred_path = ARTIFACTS_DIR / "holdout_predictions.csv"
    x_path = ARTIFACTS_DIR / "X_test.csv"
    y_path = ARTIFACTS_DIR / "y_test.csv"
    if not (pred_path.exists() and x_path.exists() and y_path.exists()):
        return

    preds = pd.read_csv(pred_path)
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()

    threshold = 0.5
    subgroup_cols = [c for c in ["imd_band", "age_band", "highest_education"] if c in X_test.columns]
    all_rows = []

    for col in subgroup_cols:
        for group_val, idx in X_test.groupby(col).groups.items():
            idx = list(idx)
            y_true_g = y_test.iloc[idx]
            y_proba_g = preds["y_proba"].iloc[idx]
            y_pred_g = (y_proba_g >= threshold).astype(int)
            n = len(y_true_g)
            if n < 20 or y_true_g.nunique() < 2:
                continue
            tp = int(((y_pred_g == 1) & (y_true_g == 1)).sum())
            fp = int(((y_pred_g == 1) & (y_true_g == 0)).sum())
            fn = int(((y_pred_g == 0) & (y_true_g == 1)).sum())
            all_rows.append({
                "subgroup_col": col,
                "subgroup_value": str(group_val),
                "n": n,
                "withdrawal_rate": float(y_true_g.mean()),
                "recall": float(recall_score(y_true_g, y_pred_g, zero_division=0)),
                "f1": float(f1_score(y_true_g, y_pred_g, zero_division=0)),
                "pr_auc": float(average_precision_score(y_true_g, y_proba_g)),
                "false_positive_rate": float(fp / max(fp + (n - int(y_true_g.sum())), 1)),
                "missed_withdrawals": fn,
            })

    if not all_rows:
        return

    subgroup_df = pd.DataFrame(all_rows)
    subgroup_df.to_csv(ARTIFACTS_DIR / "demographic_subgroup_report.csv", index=False)

    # Plot recall by IMD band (socioeconomic fairness)
    imd_df = subgroup_df[subgroup_df["subgroup_col"] == "imd_band"].sort_values("subgroup_value")
    if not imd_df.empty:
        plt.figure(figsize=(9, 5))
        plt.bar(imd_df["subgroup_value"], imd_df["recall"], color="#8E44AD")
        plt.title("Recall by IMD Band (Socioeconomic Group)\n"
                  "Low recall = model misses more at-risk students in that group")
        plt.xlabel("IMD Band")
        plt.ylabel("Recall (withdrawn students correctly flagged)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "subgroup_recall_by_imd.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def _permutation_importance_comparison() -> None:
    model_path = MODELS_DIR / "best_model.joblib"
    x_path = ARTIFACTS_DIR / "X_test.csv"
    y_path = ARTIFACTS_DIR / "y_test.csv"
    fi_path = ARTIFACTS_DIR / "feature_importance.csv"
    if not (model_path.exists() and x_path.exists() and y_path.exists()):
        return

    best_model = joblib.load(model_path)
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()

    result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10, random_state=42,
        scoring="average_precision", n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).head(20)
    perm_df.to_csv(ARTIFACTS_DIR / "permutation_importance.csv", index=False)

    # Side-by-side comparison with impurity importance
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        # Get raw feature names (before OHE prefix) for alignment
        perm_top = perm_df.head(15).copy()
        plt.figure(figsize=(9, 6))
        plt.barh(range(len(perm_top)), perm_top["perm_importance_mean"].values,
                 xerr=perm_top["perm_importance_std"].values,
                 color="#E74C3C", alpha=0.8, label="Permutation (PR-AUC drop)")
        plt.yticks(range(len(perm_top)), perm_top["feature"].values)
        plt.xlabel("Mean PR-AUC drop when feature shuffled")
        plt.title("Permutation Feature Importance — Top 15 (Holdout Set)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "permutation_importance.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Temporal summary
# ---------------------------------------------------------------------------

def _plot_temporal_summary(temporal_metrics: dict, comparison_df: pd.DataFrame) -> None:
    if comparison_df.empty:
        return
    best_row = comparison_df[comparison_df["model"] != "baseline_dummy"].iloc[0]
    labels = ["CV PR-AUC (5-fold random)", "Random-split holdout", "Temporal holdout (2014)"]
    values = [
        float(best_row.get("cv_pr_auc_mean", 0)),
        float(temporal_metrics.get("random_split_pr_auc", 0)),
        float(temporal_metrics.get("pr_auc", 0)),
    ]
    colors = ["#2980B9", "#27AE60", "#E74C3C"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.title("PR-AUC Across Evaluation Regimes\n"
              "(Higher temporal bar = model generalises to new cohort)")
    plt.ylabel("PR-AUC")
    plt.ylim(0, min(1.0, max(values) + 0.1))
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "temporal_holdout_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    create_evaluation_report()
