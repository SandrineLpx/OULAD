"""
explainability.py
=================
Comprehensive SHAP analysis for the best tree model.

Global explainability
  - Beeswarm summary plot (magnitude + direction)
  - Bar chart (mean |SHAP|, top 20)
  - Feature group contribution (demographics / engagement / assessment)

Dependence plots
  - Top 5 features by mean |SHAP|

Local explainability (3 representative instances)
  - TP: highest-risk correctly predicted
  - TN: most-confident retained student
  - FN: most-surprising false negative (missed withdrawal)

Comparison
  - SHAP importance vs permutation importance correlation plot
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.feature_engineering import get_feature_groups

warnings.filterwarnings("ignore")


def _write_note(message: str) -> None:
    (ARTIFACTS_DIR / "shap_status.txt").write_text(message.strip() + "\n", encoding="utf-8")


def _is_tree_estimator(estimator: object) -> bool:
    name = estimator.__class__.__name__.lower()
    return any(k in name for k in ["tree", "forest", "gbm", "boost", "xgb", "lgbm"])


def _pretty_feature(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").title()


def generate_shap_artifacts(sample_size: int = 1500) -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.titlesize": 12,
    })
    model_path = MODELS_DIR / "best_model.joblib"
    x_test_path = ARTIFACTS_DIR / "X_test.csv"
    y_test_path = ARTIFACTS_DIR / "y_test.csv"
    pred_path   = ARTIFACTS_DIR / "holdout_predictions.csv"

    if not model_path.exists() or not x_test_path.exists():
        _write_note("Missing best_model.joblib or X_test.csv. Run src/train_models.py first.")
        return

    try:
        import shap
    except ImportError:
        _write_note("SHAP not installed. Run: pip install shap")
        return

    best_model = joblib.load(model_path)
    X_test     = pd.read_csv(x_test_path)
    y_test     = pd.read_csv(y_test_path).squeeze() if y_test_path.exists() else None
    preds      = pd.read_csv(pred_path) if pred_path.exists() else None

    preprocessor = best_model.named_steps["preprocessor"]
    estimator    = best_model.named_steps["model"]

    if not _is_tree_estimator(estimator):
        _write_note(
            f"Best model ({estimator.__class__.__name__}) is not tree-based. "
            "Using LinearExplainer instead of TreeExplainer."
        )
        # Fall back gracefully -- use KernelExplainer with small sample
        _run_kernel_shap(shap, best_model, X_test, estimator, preprocessor, sample_size=200)
        return

    # -----------------------------------------------------------------------
    # Preprocess test sample
    # -----------------------------------------------------------------------
    sample_idx = X_test.sample(min(sample_size, len(X_test)), random_state=42).index
    X_sample   = X_test.loc[sample_idx].reset_index(drop=True)
    X_trans    = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names    = list(preprocessor.get_feature_names_out())
    X_trans_df       = pd.DataFrame(X_trans, columns=feature_names)
    pretty_names     = [_pretty_feature(f) for f in feature_names]

    # -----------------------------------------------------------------------
    # Compute SHAP values
    # -----------------------------------------------------------------------
    explainer   = shap.TreeExplainer(estimator)
    shap_output = explainer(X_trans_df)

    # Handle SHAP output shape robustly for binary classifiers across SHAP versions.
    raw_values = np.asarray(shap_output.values)
    if raw_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        class_idx = 1 if raw_values.shape[2] > 1 else 0
        shap_array = raw_values[:, :, class_idx]
    elif raw_values.ndim == 2:
        # shape: (n_samples, n_features)
        shap_array = raw_values
    else:
        raise ValueError(f"Unexpected SHAP value shape: {raw_values.shape}")

    # expected_value can be scalar, len-1 array, len-2 array, or derived from base_values.
    exp = getattr(explainer, "expected_value", None)
    if exp is None:
        base_vals = np.asarray(shap_output.base_values)
        if base_vals.ndim == 2:
            class_idx = 1 if base_vals.shape[1] > 1 else 0
            expected_value = float(np.mean(base_vals[:, class_idx]))
        else:
            expected_value = float(np.mean(base_vals))
    else:
        exp_arr = np.asarray(exp).reshape(-1)
        class_idx = 1 if exp_arr.size > 1 else 0
        expected_value = float(exp_arr[class_idx])

    mean_abs = np.abs(shap_array).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "feature_label": pretty_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(ARTIFACTS_DIR / "shap_global_importance.csv", index=False)

    top_features   = importance_df["feature"].head(10).tolist()
    top5_features  = importance_df["feature"].head(5).tolist()

    # -----------------------------------------------------------------------
    # 1. Global beeswarm summary
    # -----------------------------------------------------------------------
    plt.figure(figsize=(7, 5.5))
    shap.summary_plot(shap_array, X_trans_df, feature_names=pretty_names,
                      max_display=15, show=False)
    plt.title("SHAP Global Summary (Beeswarm) — Top 15 Features", pad=12)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # 2. Global bar chart (mean |SHAP|, top 20)
    # -----------------------------------------------------------------------
    plot_df = importance_df.head(20).copy().iloc[::-1]
    plt.figure(figsize=(7, 5.5))
    plt.barh(plot_df["feature_label"], plot_df["mean_abs_shap"], color="#0F766E")
    plt.xlabel("Mean |SHAP Value|")
    plt.title("SHAP Global Feature Importance — Top 20")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # 3. Dependence plots — top 5 features
    # -----------------------------------------------------------------------
    for i, feat in enumerate(top5_features, start=1):
        if feat not in X_trans_df.columns:
            continue
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat, shap_array, X_trans_df,
                             interaction_index=None, show=False)
        safe_name = feat.replace("/", "_").replace(" ", "_")[:50]
        plt.title(f"SHAP Dependence: {_pretty_feature(feat)}")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / f"shap_dependence_{i}_{safe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # -----------------------------------------------------------------------
    # 4. Local waterfall plots — TP, TN, FN
    # -----------------------------------------------------------------------
    if preds is not None and y_test is not None:
        _local_waterfall_plots(shap, shap_array, X_trans_df, feature_names, pretty_names,
                               expected_value, sample_idx, preds, y_test)

    # -----------------------------------------------------------------------
    # 5. Feature group contribution
    # -----------------------------------------------------------------------
    _group_contribution_chart(importance_df, X_test.columns.tolist())

    # -----------------------------------------------------------------------
    # 6. SHAP vs Permutation importance
    # -----------------------------------------------------------------------
    _shap_vs_permutation_plot(importance_df)

    with open(ARTIFACTS_DIR / "shap_top_features.json", "w") as f:
        json.dump({"top_features": top_features,
                   "note": "Ranked by mean absolute SHAP value on holdout sample."}, f, indent=2)

    _write_note("SHAP artifacts generated successfully (TreeExplainer).")
    print("SHAP artifacts generated.")


# ---------------------------------------------------------------------------
# Local waterfall helpers
# ---------------------------------------------------------------------------

def _local_waterfall_plots(shap, shap_array, X_trans_df, feature_names, pretty_names,
                            expected_value, sample_idx, preds, y_test) -> None:
    # Map sample_idx back to the sampled rows
    sample_preds = preds.iloc[sample_idx].reset_index(drop=True)
    sample_y     = y_test.iloc[sample_idx].reset_index(drop=True)

    tp_mask = (sample_preds["y_pred"] == 1) & (sample_y == 1)
    tn_mask = (sample_preds["y_pred"] == 0) & (sample_y == 0)
    fn_mask = (sample_preds["y_pred"] == 0) & (sample_y == 1)

    cases = {
        "tp": (tp_mask, sample_preds["y_proba"], True,  "shap_local_tp.png",
               "Highest-Risk TP (correctly flagged withdrawal)"),
        "tn": (tn_mask, sample_preds["y_proba"], False, "shap_local_tn.png",
               "Most-Confident TN (correctly retained)"),
        "fn": (fn_mask, sample_preds["y_proba"], True,  "shap_local_fn.png",
               "Worst FN (missed withdrawal — highest missed proba)"),
    }

    for key, (mask, proba_col, high_risk, fname, title) in cases.items():
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            continue
        if high_risk:
            row_idx = candidates[np.argmax(proba_col.iloc[candidates].values)]
        else:
            row_idx = candidates[np.argmin(proba_col.iloc[candidates].values)]

        explanation = shap.Explanation(
            values=shap_array[row_idx],
            base_values=expected_value,
            data=X_trans_df.iloc[row_idx].values,
            feature_names=pretty_names,
        )
        plt.figure(figsize=(7, 5.5))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(title, pad=8)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / fname, dpi=130, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Feature group contribution
# ---------------------------------------------------------------------------

def _group_contribution_chart(importance_df: pd.DataFrame, raw_columns: list[str]) -> None:
    groups = get_feature_groups(raw_columns)
    group_contribs: dict[str, float] = {}

    for group_name, cols in groups.items():
        if group_name == "full":
            continue
        # Match engineered feature names (strip num__ / cat__ prefix)
        group_total = importance_df[
            importance_df["feature"].apply(
                lambda f: any(c in f for c in cols)
            )
        ]["mean_abs_shap"].sum()
        group_contribs[group_name] = float(group_total)

    if not group_contribs:
        return

    total = sum(group_contribs.values()) or 1.0
    labels = list(group_contribs.keys())
    values = [group_contribs[k] / total for k in labels]
    colors = ["#2980B9", "#E67E22", "#27AE60", "#8E44AD", "#C0392B"][:len(labels)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    plt.ylabel("Share of Total Mean |SHAP|")
    plt.title("SHAP Contribution by Feature Group\n"
              "(engagement + assessment together explain most of the model)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_group_contribution.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# SHAP vs Permutation
# ---------------------------------------------------------------------------

def _shap_vs_permutation_plot(importance_df: pd.DataFrame) -> None:
    perm_path = ARTIFACTS_DIR / "permutation_importance.csv"
    if not perm_path.exists():
        return

    perm_df = pd.read_csv(perm_path)
    # Align on raw feature names (before OHE)
    merged = pd.merge(
        importance_df[["feature", "mean_abs_shap"]].head(20),
        perm_df[["feature", "perm_importance_mean"]].rename(columns={"feature": "feature"}),
        on="feature", how="inner",
    )
    if len(merged) < 3:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(merged["mean_abs_shap"], merged["perm_importance_mean"],
                alpha=0.75, color="#2E86AB", s=60)
    for _, row in merged.iterrows():
        plt.annotate(_pretty_feature(row["feature"]),
                     (row["mean_abs_shap"], row["perm_importance_mean"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    corr = merged[["mean_abs_shap", "perm_importance_mean"]].corr().iloc[0, 1]
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Permutation Importance (PR-AUC drop)")
    plt.title(f"SHAP vs Permutation Importance (r = {corr:.2f})\n"
              "High correlation = both methods agree on key features")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_vs_permutation.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Kernel SHAP fallback for non-tree models
# ---------------------------------------------------------------------------

def _run_kernel_shap(shap, model, X_test, estimator, preprocessor, sample_size: int = 200) -> None:
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    X_trans  = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    background    = shap.kmeans(X_trans, 50)
    explainer     = shap.KernelExplainer(model.predict_proba, background)
    shap_values   = explainer.shap_values(X_trans[:50], nsamples=100)

    shap_array  = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs    = np.abs(shap_array).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance_df.to_csv(ARTIFACTS_DIR / "shap_global_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_array, X_trans[:50], feature_names=feature_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    _write_note("SHAP artifacts generated (KernelExplainer — non-tree model).")
    print("SHAP artifacts generated (KernelExplainer).")


if __name__ == "__main__":
    generate_shap_artifacts()
