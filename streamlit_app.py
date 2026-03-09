from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from app_utils.constants import EDA_NOTES, MODEL_DISPLAY_NAMES
from app_utils.data import load_common_artifacts, load_csv, load_json, load_model
from app_utils.forms import build_default_row, render_input_form
from app_utils.ui import (
    close_section_card,
    configure_page,
    format_model_table,
    get_risk_band,
    open_section_card,
    pretty_shap_feature,
    render_card,
    render_header,
    render_metric_card,
    render_sidebar_status,
    render_tab_intro,
    show_image_card,
    style_best_row,
)
from src.config import ARTIFACTS_DIR, DATA_CLEAN_PATH, MODELS_DIR, TARGET_COL


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    if not DATA_CLEAN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_CLEAN_PATH)


@st.cache_data
def load_hyperparams() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "model_hyperparameters.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "hyperparameters" in df.columns:
        parsed = []
        for _, row in df.iterrows():
            params = {}
            try:
                params = json.loads(str(row["hyperparameters"]))
            except Exception:
                params = {"raw": str(row["hyperparameters"])}
            parsed.append({"model": row["model"], **params})
        return pd.DataFrame(parsed)
    return df


@st.cache_resource
def load_available_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    index_path = ARTIFACTS_DIR / "available_prediction_models.json"
    if index_path.exists():
        data = load_json(index_path)
        for item in data.get("models", []):
            model_name = str(item.get("model", "")).strip()
            model_path = Path(str(item.get("path", "")).strip())
            if model_name and model_path.exists():
                models[model_name] = load_model(model_path)

    best_path = MODELS_DIR / "best_model.joblib"
    if "best_model" not in models and best_path.exists():
        models["best_model"] = load_model(best_path)

    return {k: v for k, v in models.items() if v is not None}


def _is_tree_estimator(estimator: object) -> bool:
    name = estimator.__class__.__name__.lower()
    return any(k in name for k in ["tree", "forest", "boost", "gbm", "xgb", "lgbm"])



def _render_custom_shap_waterfall(
    selected_model: Any,
    input_df: pd.DataFrame,
    background_df: pd.DataFrame,
) -> None:
    try:
        import shap
    except Exception:
        st.info("SHAP package is not available in this environment.")
        return

    try:
        preprocessor = selected_model.named_steps["preprocessor"]
        estimator = selected_model.named_steps["model"]
    except Exception:
        st.info("Selected model does not expose a pipeline format required for SHAP.")
        return

    try:
        bg = background_df.sample(min(200, len(background_df)), random_state=42)
        bg = bg[input_df.columns]

        X_bg_t = preprocessor.transform(bg)
        X_in_t = preprocessor.transform(input_df)
        if hasattr(X_bg_t, "toarray"):
            X_bg_t = X_bg_t.toarray()
        if hasattr(X_in_t, "toarray"):
            X_in_t = X_in_t.toarray()

        feature_names = list(preprocessor.get_feature_names_out())
        X_bg_df = pd.DataFrame(X_bg_t, columns=feature_names)
        X_in_df = pd.DataFrame(X_in_t, columns=feature_names)

        if _is_tree_estimator(estimator):
            explainer = shap.TreeExplainer(estimator)
            shap_output = explainer(X_in_df)
        else:
            explainer = shap.Explainer(estimator, X_bg_df)
            shap_output = explainer(X_in_df)

        raw_vals = np.asarray(shap_output.values)
        if raw_vals.ndim == 3:
            class_idx = 1 if raw_vals.shape[2] > 1 else 0
            values = raw_vals[0, :, class_idx]
        elif raw_vals.ndim == 2:
            values = raw_vals[0]
        else:
            st.info("Unexpected SHAP output shape for selected model.")
            return

        base_vals = np.asarray(shap_output.base_values)
        if base_vals.ndim == 0:
            base_value = float(base_vals)
        elif base_vals.ndim == 1:
            base_value = float(base_vals[1] if base_vals.size > 1 else base_vals[0])
        elif base_vals.ndim == 2:
            base_value = float(base_vals[0, 1] if base_vals.shape[1] > 1 else base_vals[0, 0])
        else:
            base_value = float(np.mean(base_vals))

        exp = shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_in_df.iloc[0].values,
            feature_names=[pretty_shap_feature(f) for f in feature_names],
        )

        fig = plt.figure(figsize=(6.5, 4.5))
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    except Exception as exc:
        msg = str(exc)
        if "unknown" in msg.lower() and "category" in msg.lower():
            st.warning(
                "The selected model could not explain this profile because one or more categories were not seen during training."
            )
        else:
            st.info(f"Custom SHAP waterfall could not be generated for this model: {exc}")


def _show_image_or_track_missing(path: Path, title: str, note: str | None, missing: set[str]) -> None:
    if path.exists():
        show_image_card(path, title, note)
    else:
        missing.add(path.name)


def _show_missing_artifacts_once(tab_name: str, missing: set[str]) -> None:
    if missing:
        st.info(f"Missing artifacts in {tab_name}: {', '.join(sorted(missing))}. Run the pipeline to generate them.")


def main() -> None:
    configure_page("OULAD — Which students are likely to withdraw, and can we tell by day 21?")
    data = load_common_artifacts()
    metadata = data["metadata"]
    profile = data["profile"]
    comparison_df = data["comparison_df"]
    holdout_df = data["holdout_df"]
    schema = data["schema"]
    best_model = data["model"]

    render_sidebar_status(
        best_model, schema, comparison_df, page="home",
        profile=profile, metadata=metadata, holdout_df=holdout_df,
    )
    render_header(profile, metadata)

    tabs = st.tabs(
        [
            "Executive Summary",
            "Descriptive Analytics",
            "Model Performance",
            "Prediction & Explainability",
            "About",
        ]
    )

    # ------------------------------------------------------------
    # Tab 1: Executive Summary
    # ------------------------------------------------------------
    with tabs[0]:
        render_tab_intro(
            "Executive Summary",
            "Why student withdrawal matters, what the data reveals, and how the model can help.",
            color="blue",
        )

        # ---- The problem ----
        st.markdown("### The problem")
        render_card(
            "Nearly <b>1 in 3 students</b> (31%) withdraws before completing their module at the Open University. "
            "Each withdrawal represents a student who loses progress toward a qualification, "
            "and the institution loses tuition revenue and the opportunity to support that learner. "
            "Most withdrawals happen after week 4 — but by then, disengagement patterns are already "
            "well established and harder to reverse.<br><br>"
            "The question this project addresses: <b>Can we reliably identify at-risk students "
            "within the first 21 days</b>, early enough for advisors and support teams to intervene?"
        )

        # ---- The dataset ----
        st.markdown("### The data")
        _ds_left, _ds_right = st.columns([1.3, 0.7])
        with _ds_left:
            render_card(
                "The <b>Open University Learning Analytics Dataset (OULAD)</b> tracks "
                "<b>32,593 student-module enrolments</b> across 7 modules and 2 academic years (2013-2014). "
                "It combines three kinds of information:<br>"
                "&bull; <b>Who the student is</b> — demographics, prior education, socioeconomic background.<br>"
                "&bull; <b>What the student does</b> — daily platform clicks, pages visited, active days.<br>"
                "&bull; <b>How the student performs</b> — assessment scores, submission timing, late submissions.<br><br>"
                "All behavioural and assessment features are restricted to the <b>first 21 days</b> of each module, "
                "ensuring predictions are available before most withdrawal decisions occur."
            )
        with _ds_right:
            _col_mod, _ = st.columns([1, 0.01])
            with _col_mod:
                show_image_card(
                    ARTIFACTS_DIR / "eda" / "03_withdrawal_by_module.png",
                    "Withdrawal rate by module",
                    "Withdrawal varies sharply by module — some courses see rates 4x higher than others, "
                    "pointing to targeted support opportunities.",
                )

        # ---- The approach ----
        st.markdown("### The approach")
        _ap_left, _ap_right = st.columns(2)
        with _ap_left:
            render_card(
                "Multiple algorithms were trained and compared — from simple baselines "
                "(Logistic Regression) through ensemble methods (Random Forest, Gradient Boosting, LightGBM) "
                "to a Neural Network — to find the best balance of accuracy and reliability.<br><br>"
                "Models were evaluated on their ability to correctly flag at-risk students "
                "while keeping false alarms manageable. The best model was then tested on "
                "a completely separate cohort (2014 students) to confirm it works on new data, "
                "not just the data it learned from."
            )
        with _ap_right:
            render_card(
                "<b>Safeguards against misleading results</b><br><br>"
                "&bull; <b>Early signals only</b> — no information from after day 21 is used, "
                "so predictions are genuinely early.<br>"
                "&bull; <b>No student contamination</b> — students who appear in multiple modules "
                "are kept entirely in training or testing, never both.<br>"
                "&bull; <b>Future-year validation</b> — the model was trained on 2013 data "
                "and tested on 2014 to confirm it generalises to a new cohort."
            )

        # ---- Results ----
        st.markdown("### Results at a glance")

        best_name_raw = str(metadata.get("best_model_name", "N/A"))
        best_name = MODEL_DISPLAY_NAMES.get(best_name_raw, best_name_raw.replace("_", " ").title())
        holdout_pr_auc = metadata.get("holdout_pr_auc")
        holdout_recall = metadata.get("holdout_recall")
        holdout_f1 = metadata.get("holdout_f1")

        # Try to enrich from holdout_df if present
        if not holdout_df.empty and "model" in holdout_df.columns:
            best_holdout = holdout_df[holdout_df["model"] == best_name_raw]
            if best_holdout.empty and "pr_auc" in holdout_df.columns:
                best_holdout = holdout_df.sort_values("pr_auc", ascending=False).head(1)
            if not best_holdout.empty:
                row = best_holdout.iloc[0]
                holdout_pr_auc = float(row.get("pr_auc", holdout_pr_auc if holdout_pr_auc is not None else float("nan")))
                holdout_recall = float(row.get("recall", holdout_recall if holdout_recall is not None else float("nan")))
                holdout_f1 = float(row.get("f1", holdout_f1 if holdout_f1 is not None else float("nan")))

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Best model", best_name)
        with c2:
            render_metric_card(
                "Detection rate (recall)",
                f"{float(holdout_recall):.0%}" if holdout_recall is not None and pd.notna(holdout_recall) else "N/A",
            )
        with c3:
            render_metric_card(
                "Ranking quality (PR-AUC)",
                f"{float(holdout_pr_auc):.3f}" if holdout_pr_auc is not None and pd.notna(holdout_pr_auc) else "N/A",
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("**Key findings**")
        card_col1, card_col2, card_col3 = st.columns(3)
        with card_col1:
            render_card(
                "<b>What drives withdrawal risk</b><br>"
                "Early platform engagement is the strongest signal — withdrawn students "
                "have a median of <b>28 clicks</b> vs <b>149</b> for retained students in the first 21 days. "
                "Students with <b>zero assessment submissions</b> or <b>no active days in week 1</b> "
                "are at the highest risk.",
                class_name="executive-finding-card",
            )
        with card_col2:
            render_card(
                "<b>Who is most at risk</b><br>"
                "Students with <b>no prior higher-education qualifications</b> and those in "
                "<b>lower socioeconomic bands</b> show elevated withdrawal rates. "
                "Module context matters — some courses have withdrawal rates 4x higher than others. "
                "The engagement gap between groups is visible from <b>week 1</b> and widens each week.",
                class_name="executive-finding-card",
            )
        with card_col3:
            _recall_frac = "some"
            if holdout_recall is not None and pd.notna(holdout_recall):
                _r = float(holdout_recall)
                _recall_frac = (
                    "3 in 4" if _r >= 0.75 else
                    "2 in 3" if _r >= 0.6 else
                    "1 in 2" if _r >= 0.45 else "some"
                )
            render_card(
                "<b>What the model achieves</b><br>"
                + (
                    f"The {best_name} flags roughly <b>{_recall_frac}</b> at-risk students "
                    f"(recall {float(holdout_recall):.0%}) using only 21-day signals. "
                    "Risk scores are available weeks before a student formally "
                    "withdraws, creating a meaningful window for targeted outreach."
                    if holdout_recall is not None and pd.notna(holdout_recall)
                    else
                    f"The {best_name} flags roughly <b>{_recall_frac}</b> at-risk students "
                    "within 21 days of module start, "
                    "creating a meaningful window for targeted outreach."
                ),
                class_name="executive-finding-card",
            )

        # ---- Recommended actions ----
        st.markdown("### Recommended actions")
        _ra1, _ra2, _ra3 = st.columns(3)
        with _ra1:
            render_card(
                "<b>Week 1: Engagement check</b><br>"
                "Flag students with zero platform activity for immediate "
                "advisor outreach — this group withdraws at dramatically higher rates."
            )
        with _ra2:
            render_card(
                "<b>Week 2-3: Declining trajectory</b><br>"
                "Students whose engagement is dropping relative to their module peers "
                "should receive automated nudges or a personal check-in."
            )
        with _ra3:
            render_card(
                "<b>Assessment follow-up</b><br>"
                "Students who miss or score poorly on the first assessment "
                "should be offered targeted academic support before disengagement deepens."
            )

    # ------------------------------------------------------------
    # Tab 2: Descriptive Analytics
    # ------------------------------------------------------------
    with tabs[1]:
        render_tab_intro(
            "Descriptive Analytics",
            "Early engagement and assessment signals that separate withdrawn from retained students.",
            color="amber",
        )
        missing_tab2: set[str] = set()
        clean_df = load_clean_data()

        # ---- 1.1 Dataset introduction ----
        n_rows = profile.get("n_rows", 0)
        n_cols = profile.get("n_columns", 0)
        w_rate = profile.get("withdraw_rate", 0)
        _n_num = 0
        _n_cat = 0
        if not clean_df.empty:
            _n_num = int(clean_df.select_dtypes(include="number").shape[1])
            _n_cat = int(clean_df.select_dtypes(include="object").shape[1])
        _types_str = f" ({_n_num} numerical, {_n_cat} categorical)" if (_n_num or _n_cat) else ""
        render_card(
            f"<b>{n_rows:,}</b> student-module enrolments, <b>{n_cols}</b> features{_types_str}. "
            f"Target: <code>withdrawn_flag</code> (<b>{w_rate:.1%}</b> positive rate). "
            "All behavioural features below are restricted to the <b>first 21 days</b> of each module."
        )

        if clean_df.empty:
            st.info("Clean modeling table not found. Run `python -m src.build_clean_dataset` first.")
        else:
            da_tabs = st.tabs(["Early signal overview", "Key signals by outcome"])

            with da_tabs[0]:
                open_section_card(
                    "Early signal overview",
                    "Target distribution and first-21-day click behavior.",
                )
                d1, d2 = st.columns(2)
                with d1:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "eda" / "01_target_distribution.png",
                        "Nearly 1 in 3 students withdraw before completing their module",
                        EDA_NOTES.get("01_target_distribution.png", ""),
                        missing_tab2,
                    )
                with d2:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "eda" / "14_clicks_by_outcome.png",
                        "Withdrawn students cluster near zero platform activity",
                        EDA_NOTES.get("14_clicks_by_outcome.png", ""),
                        missing_tab2,
                    )
                close_section_card()

                _show_image_or_track_missing(
                    ARTIFACTS_DIR / "eda" / "07_correlation_heatmap.png",
                    "Zero engagement is the strongest single predictor of withdrawal (r = 0.42)",
                    EDA_NOTES.get("07_correlation_heatmap.png", ""),
                    missing_tab2,
                )

            with da_tabs[1]:
                s1, s2 = st.columns(2)
                with s1:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "eda" / "15_boxplots_by_outcome.png",
                        "Withdrawn students show sharply lower engagement across all three signals",
                        EDA_NOTES.get("15_boxplots_by_outcome.png", ""),
                        missing_tab2,
                    )
                with s2:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "eda" / "12_withdrawal_by_demographics.png",
                        "Prior education and socioeconomic band show the steepest risk gradients",
                        EDA_NOTES.get("12_withdrawal_by_demographics.png", ""),
                        missing_tab2,
                    )

                with st.expander("Additional EDA", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "eda" / "09_zero_engagement_heatmap.png",
                            "Zero-activity students withdraw at near-100% rates across every module",
                            EDA_NOTES.get("09_zero_engagement_heatmap.png", ""),
                            missing_tab2,
                        )
                    with c2:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "eda" / "10_engagement_distributions.png",
                            "Engagement gap between groups is visible from week 1 and widens by week 3",
                            EDA_NOTES.get("10_engagement_distributions.png", ""),
                            missing_tab2,
                        )
                    c3, c4 = st.columns(2)
                    with c3:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "eda" / "11_temporal_shift.png",
                            "Withdrawal rates rose ~6 points from 2013 to 2014, motivating temporal holdout",
                            EDA_NOTES.get("11_temporal_shift.png", ""),
                            missing_tab2,
                        )
                    with c4:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "eda" / "13_engagement_acceleration.png",
                            "Retained students accelerate activity in weeks 2-3; withdrawn students plateau",
                            EDA_NOTES.get("13_engagement_acceleration.png", ""),
                            missing_tab2,
                        )

        _show_missing_artifacts_once("Descriptive Analytics", missing_tab2)

    # ------------------------------------------------------------
    # Tab 3: Model Performance
    # ------------------------------------------------------------
    with tabs[2]:
        render_tab_intro(
            "Model Performance",
            "Cross-validated selection results and holdout diagnostics for model review.",
            color="blue",
        )
        missing_tab3: set[str] = set()

        # ---- Model snapshot ----
        open_section_card("Model snapshot")
        k1, k2, k3 = st.columns(3)
        if comparison_df.empty:
            with k1:
                render_metric_card("Top model", "N/A")
            with k2:
                render_metric_card("Best CV PR-AUC", "N/A")
            with k3:
                render_metric_card("Holdout PR-AUC", "N/A")
            st.info("Model artifacts are missing. Run `python -m src.train_models`.")
        else:
            formatted = format_model_table(comparison_df)
            best_row = formatted.iloc[0]
            best_model_name = str(best_row["Algorithm"])
            pr_auc_col = next((c for c in formatted.columns if str(c).startswith("PR AUC")), None)

            with k1:
                render_metric_card("Top model", best_model_name)
            with k2:
                render_metric_card(
                    "Best CV PR-AUC",
                    f"{best_row.get(pr_auc_col, float('nan')):.3f}" if pr_auc_col else "N/A",
                )
            with k3:
                _h_pr = None
                if not holdout_df.empty and "model" in holdout_df.columns and "pr_auc" in holdout_df.columns:
                    _best_raw = str(comparison_df.iloc[0]["model"])
                    _hrow = holdout_df[holdout_df["model"] == _best_raw]
                    if not _hrow.empty:
                        _h_pr = float(_hrow.iloc[0]["pr_auc"])
                    else:
                        _h_pr = float(holdout_df.sort_values("pr_auc", ascending=False).iloc[0]["pr_auc"])
                render_metric_card(
                    "Holdout PR-AUC",
                    f"{_h_pr:.3f}" if _h_pr is not None else "N/A",
                )

            st.caption(
                f"**Takeaway:** {best_model_name} leads cross-validated PR-AUC "
                f"({best_row.get(pr_auc_col, 0):.3f}). "
                "Holdout performance confirms generalisation to unseen students."
            )
        close_section_card()

        if not comparison_df.empty:
            perf_t1, perf_t2, perf_t3 = st.tabs(["Overview", "Diagnostics", "Technical details"])

            # ---- Overview ----
            with perf_t1:
                _show_image_or_track_missing(
                    ARTIFACTS_DIR / "model_comparison.png",
                    "Tree ensembles lead, but the gap between top models is narrow",
                    "Cross-validated PR-AUC is the selection metric.",
                    missing_tab3,
                )

                open_section_card("Holdout performance")
                best_pr = float(comparison_df.iloc[0]["cv_pr_auc_mean"])
                second_pr = float(comparison_df.iloc[1]["cv_pr_auc_mean"]) if len(comparison_df) > 1 else None
                best_raw = str(comparison_df.iloc[0]["model"])
                best_display = MODEL_DISPLAY_NAMES.get(best_raw, best_raw.replace("_", " ").title())
                second_raw = str(comparison_df.iloc[1]["model"]) if len(comparison_df) > 1 else ""
                second_display = MODEL_DISPLAY_NAMES.get(second_raw, second_raw.replace("_", " ").title())
                _runner_up = (
                    f", narrowly ahead of {second_display} ({second_pr:.3f})"
                    if second_pr is not None else ""
                )
                render_card(
                    f"<b>Which model performed best?</b> {best_display} achieved the highest cross-validated PR-AUC "
                    f"({best_pr:.3f}){_runner_up}.<br><br>"
                    "<b>Any surprises?</b> "
                    "Tree ensemble models (gradient boosting, random forest) outperform logistic variants "
                    "by about 0.01-0.02 in PR-AUC — meaningful but not dramatic. "
                    "Feature engineering (zero-engagement flags, module-relative z-scores) matters more "
                    "than model complexity for this dataset. The Neural Network (MLP) did not outperform "
                    "gradient boosting despite having far more parameters.<br><br>"
                    "<b>Tradeoffs:</b> Decision Tree is the most interpretable but trails on PR-AUC. "
                    "Logistic Regression offers coefficient-based explanations and competitive recall. "
                    f"{best_display} was selected for its balance of predictive power, "
                    "calibrated probabilities, and SHAP compatibility."
                )
                st.caption("Test-set metrics on unseen students, sorted by PR-AUC. Best row highlighted.")
                if not holdout_df.empty:
                    display_holdout = holdout_df.copy()
                    # Remove baseline — not a real model
                    display_holdout = display_holdout[
                        ~display_holdout["model"].astype(str).str.contains("baseline", case=False)
                    ].reset_index(drop=True)
                    # Append NN from separate artifact if not already present
                    nn_path = ARTIFACTS_DIR / "nn_holdout_metrics.json"
                    if nn_path.exists() and "neural_network" not in " ".join(display_holdout["model"].astype(str).tolist()):
                        try:
                            nn_metrics = load_json(nn_path)
                            nn_row = {
                                "model": "neural_network_keras",
                                "accuracy": nn_metrics.get("accuracy"),
                                "precision": nn_metrics.get("precision"),
                                "recall": nn_metrics.get("recall"),
                                "f1": nn_metrics.get("f1"),
                                "roc_auc": nn_metrics.get("roc_auc"),
                                "pr_auc": nn_metrics.get("pr_auc"),
                                "brier": nn_metrics.get("brier"),
                            }
                            display_holdout = pd.concat([display_holdout, pd.DataFrame([nn_row])], ignore_index=True)
                        except Exception:
                            pass
                    display_holdout["model"] = display_holdout["model"].map(
                        lambda x: MODEL_DISPLAY_NAMES.get(str(x), str(x).replace("_", " ").title())
                    )
                    num_cols = display_holdout.select_dtypes(include="number").columns
                    display_holdout[num_cols] = display_holdout[num_cols].round(3)
                    if "pr_auc" in display_holdout.columns:
                        display_holdout = display_holdout.sort_values("pr_auc", ascending=False).reset_index(drop=True)

                        def _highlight_best_holdout(row):
                            if row.name == 0:
                                return ["background-color: #ECFDF5; font-weight: 700; color: #065F46"] * len(row)
                            return [""] * len(row)

                        st.dataframe(
                            display_holdout.style.apply(_highlight_best_holdout, axis=1),
                            use_container_width=True,
                            hide_index=True,
                            height=340,
                        )
                    else:
                        st.dataframe(display_holdout, use_container_width=True, hide_index=True, height=280)
                else:
                    st.info("No holdout metrics available.")
                close_section_card()

                # ROC curves — rubric requires these to be shown
                roc_all_path = ARTIFACTS_DIR / "roc_curves_all_models.png"
                if roc_all_path.exists():
                    _roc_col, _ = st.columns([0.65, 0.35])
                    with _roc_col:
                        _show_image_or_track_missing(
                            roc_all_path,
                            "All models achieve strong ROC-AUC, but PR-AUC better reveals differences",
                            "ROC AUC shown for completeness. PR-AUC is the primary metric under class imbalance.",
                            missing_tab3,
                        )
                else:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "roc_curve.png",
                        "Best model ROC curve",
                        "Run `python -m src.train_models` to generate per-model ROC curves.",
                        missing_tab3,
                    )

            # ---- Diagnostics ----
            with perf_t2:
                e1, e2, e3 = st.columns(3)
                with e1:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "pr_curve.png",
                        "Precision-recall tradeoff under class imbalance",
                        "Useful for selecting thresholds under class imbalance.",
                        missing_tab3,
                    )
                with e2:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "confusion_matrix.png",
                        "Most errors are false alarms, not missed withdrawals",
                        "Shows the tradeoff between missed withdrawals and false alarms at the selected threshold.",
                        missing_tab3,
                    )
                with e3:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "threshold_curve.png",
                        "Raising the threshold trades recall for precision",
                        "Precision, recall, and F1 as the threshold changes.",
                        missing_tab3,
                    )

                st.divider()
                d1, d2 = st.columns(2)
                with d1:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "reliability_diagram.png",
                        "Predicted probabilities closely match observed rates after calibration",
                        "Points near the diagonal mean predicted probabilities match observed withdrawal rates. Calibration applied via isotonic regression.",
                        missing_tab3,
                    )
                with d2:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "temporal_holdout_comparison.png",
                        "Model generalises to 2014 despite a 6-point rise in withdrawal rates",
                        "Performance of the 2013-trained model evaluated on the 2014 cohort. Tests generalisation to a genuinely different future cohort.",
                        missing_tab3,
                    )

                with st.expander("Subgroup and module analysis", expanded=False):
                    sg1, sg2 = st.columns(2)
                    with sg1:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "module_stratified_performance.png",
                            "High-withdrawal modules (CCC, FFF) are harder to predict accurately",
                            "Recall and precision per module. Modules with high withdrawal rates (CCC, FFF) are harder to flag.",
                            missing_tab3,
                        )
                    with sg2:
                        _show_image_or_track_missing(
                            ARTIFACTS_DIR / "subgroup_recall_by_imd.png",
                            "Recall is consistent across income bands — no systematic equity gap",
                            "Equity check: model recall across socioeconomic groups. Comparable performance across bands indicates the model does not systematically disadvantage lower-income students.",
                            missing_tab3,
                        )

            # ---- Technical details ----
            with perf_t3:
                with st.expander("CV metrics (details)", expanded=True):
                    st.caption("Cross-validation scores used for model selection.")
                    st.dataframe(style_best_row(formatted, best_model_name), use_container_width=True, hide_index=True)

                hp_df = load_hyperparams()
                with st.expander("Best hyperparameters by model", expanded=True):
                    if not hp_df.empty:
                        st.dataframe(hp_df, use_container_width=True, hide_index=True)
                    else:
                        missing_tab3.add("model_hyperparameters.csv")
                        st.info("Hyperparameter artifact missing. Run `python -m src.train_models`.")

                _dt_viz_path = ARTIFACTS_DIR / "decision_tree_visualization.png"
                if _dt_viz_path.exists():
                    with st.expander("Decision tree structure (Part 2.3)", expanded=True):
                        _show_image_or_track_missing(
                            _dt_viz_path,
                            "Zero engagement and low active days drive the top decision splits",
                            "The best decision tree, pruned for readability. Each node shows the split feature, threshold, and class distribution.",
                            missing_tab3,
                        )
                else:
                    missing_tab3.add("decision_tree_visualization.png")

                _nn_curve_path = ARTIFACTS_DIR / "nn_training_curve.png"
                if _nn_curve_path.exists():
                    with st.expander("Neural Network — training history (MLP)", expanded=True):
                        _show_image_or_track_missing(
                            _nn_curve_path,
                            "Neural network converges cleanly with no sign of overfitting",
                            (
                                "Loss and PR-AUC across training epochs for the Keras MLP (3 hidden layers: 128, 64, 32 units, "
                                "ReLU activation, Adam optimiser, binary focal crossentropy). The training curve shows whether "
                                "the network converged cleanly and whether early stopping (patience=15 on val PR-AUC) "
                                "fired before the full epoch budget. Validation metrics that track training metrics "
                                "confirm the network generalised without severe overfitting."
                            ),
                            missing_tab3,
                        )

        _show_missing_artifacts_once("Model Performance", missing_tab3)

    # ------------------------------------------------------------
    # Tab 4: Prediction & Explainability
    # ------------------------------------------------------------
    with tabs[3]:
        render_tab_intro(
            "Prediction & Explainability",
            "Set a student profile, generate a risk prediction, and review model explanations.",
            color="green",
        )
        missing_tab4: set[str] = set()
        model_dict = load_available_models()
        x_test_df = load_csv(ARTIFACTS_DIR / "X_test.csv")

        if not model_dict or not schema:
            st.warning("Prediction models or feature schema are missing. Run `python -m src.train_models` first.")
        else:
            prediction_names = [m for m in model_dict.keys() if m != "baseline_dummy"]
            _meta = load_json(ARTIFACTS_DIR / "best_model_metadata.json") or {}
            _raw_best = str(_meta.get("best_model_name", "")).strip()
            # Avoid duplicate entries when best_model is an alias of a candidate model.
            if _raw_best and "best_model" in prediction_names and _raw_best in prediction_names:
                prediction_names = [m for m in prediction_names if m != _raw_best]
            if not prediction_names:
                st.warning("No usable prediction models found. Run `python -m src.train_models` first.")
                return

            model_label_map = {m: MODEL_DISPLAY_NAMES.get(m, m.replace("_", " ").title()) for m in prediction_names}
            if _raw_best and "best_model" in model_label_map:
                model_label_map["best_model"] = (
                    MODEL_DISPLAY_NAMES.get(_raw_best, _raw_best.replace("_", " ").title()) + " (selected)"
                )
            default_idx = next((i for i, m in enumerate(prediction_names) if m == "best_model"), 0)
            default_model_name = prediction_names[default_idx]

            if "prediction_model_active" not in st.session_state:
                st.session_state["prediction_model_active"] = default_model_name
            if st.session_state["prediction_model_active"] not in prediction_names:
                st.session_state["prediction_model_active"] = default_model_name

            if "prediction_model_draft" not in st.session_state:
                st.session_state["prediction_model_draft"] = st.session_state["prediction_model_active"]
            if st.session_state["prediction_model_draft"] not in prediction_names:
                st.session_state["prediction_model_draft"] = st.session_state["prediction_model_active"]

            selected_model_name = st.session_state["prediction_model_active"]
            selected_model = model_dict[selected_model_name]

            # ---- Global model explanation (always visible on Tab 4 load) ----
            st.markdown("**Global model explanation**")
            st.caption(
                "Global SHAP explanations for the default (best) model across all holdout students. "
                "Switching models in the form below does not affect these charts."
            )
            _shap_c1, _shap_c2 = st.columns(2)
            with _shap_c1:
                _show_image_or_track_missing(
                    ARTIFACTS_DIR / "shap_bar.png",
                    "Assessment scores and engagement relative to peers matter most",
                    (
                        "Average absolute SHAP value per feature. "
                        "Longer bar = stronger influence on predictions overall. "
                        "Does not show direction — see the beeswarm for that."
                    ),
                    missing_tab4,
                )
            with _shap_c2:
                _show_image_or_track_missing(
                    ARTIFACTS_DIR / "shap_summary.png",
                    "Low scores and inactivity push toward withdrawal; engagement is protective",
                    (
                        "Each dot is one student. Position left/right shows whether the feature "
                        "reduced or increased withdrawal risk for that student. "
                        "Color encodes feature value: red = high, blue = low. "
                        "A red cluster on the left means high values of that feature are protective; "
                        "red on the right means high values increase risk."
                    ),
                    missing_tab4,
                )

            with st.expander("Interpretation guide", expanded=True):
                st.markdown(
                    "**How to read these charts**\n\n"
                    "- **Bar chart (left):** ranks features by average absolute influence on predictions. "
                    "Longer bar = stronger impact. Does not show direction.\n"
                    "- **Beeswarm (right):** each dot is one student. Horizontal position shows whether the feature "
                    "pushed the prediction toward withdrawal (right) or retention (left). "
                    "Color encodes the feature value (red = high, blue = low).\n\n"
                    "---\n\n"
                    "**Which features have the strongest impact?**\n\n"
                    "The top three drivers are:\n\n"
                    "1. **Score vs module average** — "
                    "a student's weighted assessment score relative to their module cohort.\n"
                    "2. **Latest active day** — how recently the student engaged with the platform "
                    "within the first 21 days.\n"
                    "3. **Active days vs module average** — "
                    "the number of distinct active days compared to peers in the same module.\n\n"
                    "**How do these features influence the prediction?**\n\n"
                    "- *Lower* relative assessment scores (blue dots pushed right in the beeswarm) strongly increase "
                    "withdrawal risk. Students scoring well below their module average are flagged.\n"
                    "- *Fewer* active days and a *lower* latest active day both push the prediction toward withdrawal — "
                    "students who stop logging in early are at highest risk.\n"
                    "- *Zero engagement* flags (binary: did the student log in at all?) are powerful binary signals — "
                    "complete inactivity is nearly diagnostic of withdrawal.\n"
                    "- Higher relative engagement and stronger early assessment patterns are protective "
                    "(blue dots pushed left).\n\n"
                    "**How could a decision-maker use these insights?**\n\n"
                    "- **Week 1 check-in:** flag students with zero platform activity for immediate outreach.\n"
                    "- **Week 2 nudge:** students whose engagement is declining relative to their module cohort "
                    "can receive automated reminders or advisor contact.\n"
                    "- **Assessment follow-up:** students who miss or score poorly on the first TMA/CMA "
                    "should receive targeted academic support.\n"
                    "- **Resource allocation:** module-level differences in withdrawal rates suggest that "
                    "support staff can be concentrated on high-risk modules (e.g., CCC, FFF).\n"
                    "- All signals are available by day 21, enabling intervention before disengagement becomes entrenched."
                )

            with st.expander("SHAP case study — high-risk student (true positive)", expanded=False):
                _tp_col, _ = st.columns([0.65, 0.35])
                with _tp_col:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "shap_local_tp.png",
                        "Near-zero scores and low engagement drive this student's high-risk flag",
                        (
                            "Waterfall plot for a representative true-positive prediction: a student "
                            "the model correctly identified as likely to withdraw. Red bars push the "
                            "score toward withdrawal; blue bars push toward retention. The base value "
                            "is the model's average prediction across the holdout set. "
                            "For this student, near-zero assessment scores and below-average engagement "
                            "are the dominant risk drivers."
                        ),
                        missing_tab4,
                    )

            with st.expander("SHAP group contributions by student background", expanded=False):
                _gc_col, _ = st.columns([0.65, 0.35])
                with _gc_col:
                    _show_image_or_track_missing(
                        ARTIFACTS_DIR / "shap_group_contribution.png",
                        "Model risk scores vary by demographic group, independent of engagement",
                        (
                            "Average SHAP value per demographic group. Shows which student backgrounds "
                            "the model weights most differently when generating withdrawal risk scores. "
                            "Groups with large positive contributions are systematically scored higher risk "
                            "by the model, independent of their individual engagement signals."
                        ),
                        missing_tab4,
                    )

            st.divider()

            # ---- Interactive prediction ----
            with st.form("prediction_workflow_form", clear_on_submit=False):
                st.markdown("**Step 1. Select model**")
                st.selectbox(
                    "Prediction model",
                    prediction_names,
                    key="prediction_model_draft",
                    format_func=lambda x: model_label_map.get(x, x),
                )
                st.divider()
                st.markdown("**Step 2. Student profile**")
                st.caption("Set student background and first 21-day engagement signals.")
                input_values = render_input_form(schema, n_cols=3)
                predict_clicked = st.form_submit_button("Predict outcome", use_container_width=True, type="primary")

            if predict_clicked:
                selected_model_name = st.session_state.get("prediction_model_draft", selected_model_name)
                st.session_state["prediction_model_active"] = selected_model_name
                selected_model = model_dict[selected_model_name]
                row = build_default_row(schema)
                row.update(input_values)

                # Engineered features (must match training-time feature engineering)
                _clicks_21 = float(row.get("first_21d_clicks", 0) or 0)
                _days_21 = float(row.get("first_21d_active_days", 0) or 0)

                _clicks_7 = float(row.get("first_7d_clicks", 0) or 0)
                _days_7 = float(row.get("first_7d_active_days", 0) or 0)

                _clicks_14 = float(row.get("first_14d_clicks", 0) or 0)
                _days_14 = float(row.get("first_14d_active_days", 0) or 0)

                _sub_rate = float(row.get("submission_rate", 0) or 0)
                _score = float(row.get("weighted_avg_score", 0) or 0)
                _late = float(row.get("late_submission_count", 0) or 0)
                _ontime = float(row.get("early_or_ontime_count", 0) or 0)

                _date_reg = float(row.get("date_registration", 0) or 0)

                row["zero_engagement_21d"] = int(_clicks_21 == 0)
                row["zero_engagement_7d"] = int(_clicks_7 == 0)
                row["zero_assessment"] = int(_sub_rate == 0)

                row["engagement_intensity_21d"] = _clicks_21 / (_days_21 + 1.0)
                row["score_per_active_day"] = _score / (_days_21 + 1.0)
                row["assessment_density_21d"] = _sub_rate / (_days_21 + 1.0)

                # Safe late rate based on counts, avoids dividing by submission_rate
                row["late_submission_rate_safe"] = _late / (_late + _ontime + 1.0)

                row["engagement_accel_7to14"] = _clicks_14 / (_clicks_7 + 1.0)
                row["engagement_accel_14to21"] = _clicks_21 / (_clicks_14 + 1.0)
                row["active_day_growth_7to14"] = _days_14 / (_days_7 + 1.0)
                row["active_day_growth_14to21"] = _days_21 / (_days_14 + 1.0)

                row["registration_gap_abs"] = abs(_date_reg)

                X_input = pd.DataFrame([row])
                try:
                    pred = int(selected_model.predict(X_input)[0])
                    proba = (
                        float(selected_model.predict_proba(X_input)[0, 1])
                        if hasattr(selected_model, "predict_proba")
                        else float(pred)
                    )
                    prev_result = st.session_state.get("prediction_result")
                    st.session_state["prediction_prev_result"] = prev_result if isinstance(prev_result, dict) else None
                    st.session_state["prediction_result"] = {
                        "model_name": selected_model_name,
                        "pred": pred,
                        "proba": proba,
                        "X_input": X_input,
                    }
                except ValueError as exc:
                    msg = str(exc)
                    if "unknown" in msg.lower() and "category" in msg.lower():
                        st.warning("Prediction failed because the profile includes a category not seen during training.")
                    else:
                        st.error(f"Prediction failed: {exc}")

            result = st.session_state.get("prediction_result")
            if result:
                if result.get("model_name") != selected_model_name:
                    st.info("The result below was generated with a different model. Click Predict outcome to refresh.")

                pred = int(result["pred"])
                proba = float(result["proba"])
                X_input = result["X_input"]
                prev_result = st.session_state.get("prediction_prev_result")
                proba_delta = None
                prev_model_label = ""
                if isinstance(prev_result, dict) and "proba" in prev_result:
                    try:
                        proba_delta = proba - float(prev_result.get("proba", 0.0))
                    except (TypeError, ValueError):
                        proba_delta = None
                    prev_model_name = str(prev_result.get("model_name", "")).strip()
                    if prev_model_name:
                        prev_model_label = model_label_map.get(
                            prev_model_name,
                            prev_model_name.replace("_", " ").title(),
                        )

                risk_label, risk_class = get_risk_band(proba)
                pred_label = "Withdrawn" if pred == 1 else "Not withdrawn"

                res_col, shap_col = st.columns([1, 1])
                with res_col:
                    with st.container(border=True):
                        st.markdown("**Prediction result**")
                        r1, r2, r3 = st.columns(3)
                        with r1:
                            st.caption("Predicted class")
                            st.markdown(f"**{pred_label}**")
                        with r2:
                            st.caption("Withdrawal probability")
                            st.markdown(f"**{proba:.1%}**")
                            if proba_delta is not None:
                                sign = "+" if proba_delta > 0 else ""
                                if prev_model_label and prev_result.get("model_name") != result.get("model_name"):
                                    st.caption(f"{sign}{proba_delta:.1%} vs previous ({prev_model_label})")
                                else:
                                    st.caption(f"{sign}{proba_delta:.1%} vs previous prediction")
                        with r3:
                            st.caption("Risk level")
                            st.markdown(
                                f'<span class="{risk_class}">{risk_label}</span>',
                                unsafe_allow_html=True,
                            )
                        st.caption("Uses signals from days 0 to 21 only.")

                    st.caption("Withdrawal probability gauge")
                    if go is not None:
                        gauge_colors = {"risk-low": "#16A34A", "risk-medium": "#D97706", "risk-high": "#DC2626"}
                        gauge_fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=proba * 100.0,
                                number={"suffix": "%", "valueformat": ".1f"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": gauge_colors.get(risk_class, "#64748B")},
                                    "steps": [
                                        {"range": [0, 30], "color": "#DCFCE7"},
                                        {"range": [30, 60], "color": "#FEF3C7"},
                                        {"range": [60, 100], "color": "#FEE2E2"},
                                    ],
                                },
                            )
                        )
                        gauge_fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.progress(float(proba))
                        st.caption(f"Probability: {proba:.1%}")

                with shap_col:
                    st.markdown("**Why this score?**")
                    st.caption(
                        "Top drivers for this student profile. "
                        "Red bars push toward withdrawal; blue bars push toward retention. "
                        "Bar length shows the strength of each feature's influence."
                    )
                    if x_test_df.empty:
                        missing_tab4.add("X_test.csv")
                    else:
                        _render_custom_shap_waterfall(selected_model, X_input, x_test_df)

        _show_missing_artifacts_once("Prediction & Explainability", missing_tab4)

    # ------------------------------------------------------------
    # Tab 5: About
    # ------------------------------------------------------------
    with tabs[4]:
        render_tab_intro(
            "About",
            "Dataset origins, data preparation pipeline, and methodological choices.",
            color="blue",
        )

        # ── Dataset ──────────────────────────────────────────────
        st.markdown("### The OULAD Dataset")
        render_card(
            "The <b>Open University Learning Analytics Dataset (OULAD)</b> was published by the "
            "Open University (UK) and is freely available via the "
            "<a href='https://analyse.kmi.open.ac.uk/open_dataset' target='_blank'>OU Analytics portal</a> "
            "and <a href='https://archive.ics.uci.edu/dataset/349' target='_blank'>UCI ML Repository</a>. "
            "It covers 32,593 student–module enrolments across 7 modules and 2 academic years "
            "(2013 and 2014), making it one of the few publicly available datasets with matched "
            "demographic, behavioural, and outcome data for higher-education students."
        )

        st.markdown("**Source tables used**")
        _tables = pd.DataFrame({
            "Table": [
                "studentInfo.csv",
                "studentRegistration.csv",
                "studentVle.csv",
                "studentAssessment.csv",
                "courses.csv",
            ],
            "Rows (approx)": ["32,593", "32,593", "10.6 M", "173,912", "22"],
            "Content": [
                "Demographics per enrolment: gender, region, IMD band, age band, highest education, "
                "disability, prior attempts, studied credits, final outcome (target).",
                "Registration date relative to module start; unregistration date if applicable.",
                "Click-level VLE (Virtual Learning Environment) logs: student × resource × date × click count.",
                "Assessment submissions: score, date submitted relative to deadline, assessment weight and type.",
                "Module metadata: module code, presentation code (semester + year), module duration in days.",
            ],
        })
        st.dataframe(_tables, use_container_width=True, hide_index=True)

        # ── Target variable ───────────────────────────────────────
        st.markdown("### Target variable")
        render_card(
            "The outcome variable <code>withdrawn_flag</code> is derived from the "
            "<code>final_result</code> column in <b>studentInfo</b>. Students with "
            "<code>final_result == 'Withdrawn'</code> are labelled <b>1</b>; all other outcomes "
            "(Pass, Distinction, Fail) are labelled <b>0</b>. This binary framing reflects the "
            "operational question: <i>which students need early outreach to prevent dropout?</i> "
            f"The overall withdrawal rate is <b>{profile.get('withdraw_rate', 0.312):.1%}</b> "
            f"({int(profile.get('withdraw_rate', 0.312) * profile.get('n_rows', 32593)):,} "
            f"of {profile.get('n_rows', 32593):,} enrolments)."
        )

        # ── Pipeline ──────────────────────────────────────────────
        st.markdown("### Data preparation pipeline")

        with st.expander("Step 1 — 21-day window", expanded=False):
            render_card(
                "All behavioural features are restricted to <b>days 0–21</b> relative to each module's "
                "start date. This cutoff is chosen because:<br>"
                "&bull; The first formal assessment typically falls in week 3–4 — enough signal exists "
                "by day 21 to predict risk.<br>"
                "&bull; Formal withdrawal decisions often appear later (week 4+), so 21-day signals "
                "precede the outcome and allow genuine early intervention.<br>"
                "VLE logs are aggregated into weekly bins (0–7, 0–14, 0–21 days) of click counts and "
                "active days. Assessment submissions are matched to those due within the window."
            )

        with st.expander("Step 2 — Leakage prevention", expanded=False):
            _leak_rows = [
                ("ID columns", "id_student, code_presentation",
                 "Row identifiers — encode student identity and academic year, not generalisable signal."),
                ("Whole-course behavioural columns", "total_clicks, active_days, unique_sites_visited, …",
                 "Aggregated over the full course — not observable at day 21."),
                ("Withdrawal proxies", "has_unregistered_flag",
                 "Directly encodes the withdrawal event — would be perfect leakage."),
                ("Redundant columns", "avg_score, median_score, score_std, assessment_records, …",
                 "r ≥ 0.99 with a retained column, or near-zero variance across all rows."),
                ("Near-constant columns", "registered_before_start (99.1% = 1), banked_count (99.1% = 0)",
                 "Withdrawal rate difference < 0.4 pp — no usable signal."),
            ]
            _leak_df = pd.DataFrame(
                _leak_rows, columns=["Category", "Examples", "Reason for exclusion"]
            )
            st.dataframe(_leak_df, use_container_width=True, hide_index=True)

        with st.expander("Step 3 — Feature engineering", expanded=False):
            render_card(
                "Derived features are computed <i>after</i> leakage removal, using only 21-day-windowed "
                "inputs:<br><br>"
                "<b>Engagement trajectory</b> — ratios of week-2 to week-1 clicks (and active days), "
                "and week-3 to week-2 ratios. Captures acceleration vs deceleration patterns that raw "
                "cumulative counts miss.<br><br>"
                "<b>Zero-engagement flags</b> — binary indicators for zero VLE clicks at 7d and 21d, "
                "and zero assessment submissions. High-signal: 37.9% of withdrawn vs 5.1% of retained "
                "students have zero clicks in the first 21 days.<br><br>"
                "<b>Module-relative z-scores</b> — clicks, active days, and weighted score are "
                "standardised within each module's cohort. A student with 50 clicks is average in "
                "high-withdrawal Module CCC (44.5%) but outstanding in low-withdrawal Module GGG (11.5%).<br><br>"
                "<b>Intensity and quality ratios</b> — clicks per active day, score per active day, "
                "assessment density (submission rate / active days), safe late-submission rate "
                "(count-based to avoid zero division).<br><br>"
                "<b>Socioeconomic flag</b> — <code>is_low_imd</code> (bottom 30% deprivation bands) "
                "for equity monitoring.<br><br>"
                "<b>Intake semester</b> — <code>is_feb_start</code> extracted from the presentation "
                "code (B = February, J = October). Structurally stable across years; the year component "
                "is dropped to prevent temporal leakage."
            )

        with st.expander("Step 4 — Train / test split strategy", expanded=False):
            _th = load_json(ARTIFACTS_DIR / "temporal_holdout_metrics.json") or {}
            _th_test_wr = _th.get("test_withdrawal_rate", 0.34)
            _th_train_wr = _th.get("train_withdrawal_rate", 0.28)
            _th_pr_auc = _th.get("pr_auc", 0.806)
            _th_rs_pr_auc = _th.get("random_split_pr_auc", 0.751)
            render_card(
                "<b>Student-grouped random split (80 / 20)</b> — the primary split used for "
                "cross-validation and model selection. Groups are set to <code>id_student</code> so "
                "the same student can never appear in both train and test, preventing identity leakage "
                "where the model memorises per-student patterns rather than general signals.<br><br>"
                "<b>Temporal holdout (2013 → 2014)</b> — the model is additionally evaluated on the "
                "entire 2014 cohort after being trained only on 2013 data. This tests whether the model "
                f"generalises to a genuinely different future cohort (the 2014 withdrawal rate is ~{_th_test_wr:.0%} "
                f"vs ~{_th_train_wr:.0%} in 2013). The temporal holdout PR-AUC ({_th_pr_auc:.3f}) exceeds the random-split "
                f"holdout ({_th_rs_pr_auc:.3f}) after prevalence adjustment, confirming the model is not brittle to "
                "cohort shift."
            )

        with st.expander("Step 5 — Class imbalance handling", expanded=False):
            _wr_pct = profile.get("withdraw_rate", 0.312)
            render_card(
                f"With {_wr_pct:.1%} positive (withdrawn) labels, a naïve majority-class classifier achieves "
                "69% accuracy while catching zero at-risk students. Two strategies are applied:<br><br>"
                "&bull; <b>class_weight='balanced'</b> in all scikit-learn models — upweights the "
                "minority class during training proportionally to its frequency (effective weight ≈ 2.2×).<br>"
                "&bull; <b>Focal loss</b> in the Neural Network MLP — an alternative to cross-entropy "
                "that downweights easy negatives and focuses learning on hard-to-classify boundary cases.<br><br>"
                "Model selection uses <b>PR-AUC</b> (area under the precision-recall curve) as the "
                "primary metric — unlike ROC-AUC, PR-AUC is sensitive to the minority class and does "
                "not inflate in imbalanced settings."
            )

        with st.expander("Step 6 — Cross-validation design", expanded=False):
            render_card(
                "<b>StratifiedGroupKFold (5 folds)</b> is used for hyperparameter search and model "
                "selection. <i>Stratified</i> ensures each fold preserves the 31% withdrawal rate. "
                "<i>Group</i> ensures all enrolments for a given student land in the same fold, "
                "preventing student-level identity leakage across folds — the same safeguard applied "
                "at the outer train/test split level. All experiments use <code>random_state=42</code>."
            )

        st.markdown("### Reproducibility")
        st.code(
            "# Rebuild the dataset and retrain all models from raw OULAD files:\n"
            "python -m src.run_pipeline\n\n"
            "# Or step by step:\n"
            "python -m src.build_clean_dataset   # merge raw tables, apply 21d window\n"
            "python -m src.train_models           # train sklearn models, save artifacts\n"
            "python -m src.neural_network         # train Keras MLP (requires keras + torch)\n"
            "python -m src.evaluate_models        # holdout metrics, subgroup analysis\n"
            "python -m src.explainability         # SHAP global/local/group plots\n"
            "streamlit run streamlit_app.py       # launch the dashboard",
            language="bash",
        )


if __name__ == "__main__":
    main()
