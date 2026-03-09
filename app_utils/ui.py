from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_utils.constants import (
    COMPARISON_COL_LABELS,
    FRIENDLY_LABELS,
    MODEL_DISPLAY_NAMES,
)


CUSTOM_CSS = """
<style>
/* Page width and spacing */
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.0rem;
    max-width: 1280px;
}
/* Tighten Streamlit default spacing */
div.block-container { padding-top: 1.15rem; }
[data-testid="stAppViewContainer"] .main .block-container { padding-top: 1.15rem; }
div[data-testid="stVerticalBlock"] { gap: 0.65rem; }

/* Reduce top whitespace around headers */
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { margin-top: 0.25rem; margin-bottom: 0.25rem; line-height: 1.15; }
h2 { margin-top: 1.2rem; margin-bottom: 0.5rem; }
p { line-height: 1.45; }

/* Sidebar: make it feel lighter */
section[data-testid="stSidebar"] {
    background: #F8FAFC;
    border-right: 1px solid #E5E7EB;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.0rem; }

/* Cards */
.dashboard-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.0rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}

/* Section containers for charts and groups */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 1.0rem 1.1rem;
    margin: 0.75rem 0 1.0rem 0;
}

.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.82rem;
    color: #475569;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
}
.section-note {
    color: #475569;
    font-size: 0.95rem;
    margin-top: -0.3rem;
    margin-bottom: 1rem;
}
.small-muted {
    color: #64748B;
    font-size: 0.85rem;
}
/* Make captions more compact */
.stCaption, [data-testid="stCaptionContainer"] {
  margin-top: 0.15rem;
  margin-bottom: 0.35rem;
  line-height: 1.35;
}

/* Risk pills */
.risk-low, .risk-medium, .risk-high {
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.risk-low { background: #DCFCE7; color: #166534; }
.risk-medium { background: #FFEDD5; color: #9A3412; }
.risk-high { background: #FEE2E2; color: #991B1B; }

/* Tabs: slightly stronger active tab */
div[data-baseweb="tab-list"] button {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    font-weight: 600;
    color: #475569 !important;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Primary button */
.stButton > button {
  border-radius: 10px;
  padding: 0.55rem 0.9rem;
  font-weight: 700;
}

/* Predict outcome form submit button — size only; color set via .streamlit/config.toml primaryColor */
div[data-testid="stFormSubmitButton"] > button {
  font-size: 1.05rem;
  font-weight: 800;
  padding: 0.75rem 1.2rem;
  border-radius: 10px;
  letter-spacing: 0.02em;
}

/* Selectbox, slider spacing */
div[data-testid="stSelectbox"], div[data-testid="stSlider"] {
  margin-bottom: 0.35rem;
}

/* Reduce expander padding */
details[data-testid="stExpander"] summary {
  padding-top: 0.35rem;
  padding-bottom: 0.35rem;
}
details[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
  padding-top: 0.25rem;
}

/* Reduce excessive space after charts */
div[data-testid="stPlotlyChart"], div[data-testid="stPyplotFigure"] {
    margin-bottom: 0.2rem;
}
/* Make images in cards feel like dashboard tiles */
.dashboard-card img, .section-card img {
  border-radius: 10px;
}

.result-panel {
    background: #F8FAFC;
    border: 1px solid #DDE5EE;
    border-left: 5px solid #0F766E;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-top: 0.9rem;
}
.result-title {
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.6rem;
}
.result-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.result-k {
    color: #475569;
}
.result-v {
    color: #0F172A;
    font-weight: 600;
}
hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* ---- Hero banner (Home page only) ---- */
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F766E 100%);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 0.4rem;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 1rem;
    color: #CBD5E1;
    line-height: 1.6;
    max-width: 700px;
}

/* ---- Colored section header strips ---- */
.section-header-blue {
    background: #EFF6FF;
    border-left: 4px solid #1D4ED8;
    border-radius: 0 8px 8px 0;
    padding: 0.45rem 0.9rem;
    margin-bottom: 0.8rem;
    font-weight: 700;
    color: #1E3A5F;
    font-size: 1.05rem;
}
.section-header-amber {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 0.45rem 0.9rem;
    margin-bottom: 0.8rem;
    font-weight: 700;
    color: #92400E;
    font-size: 1.05rem;
}
.section-header-green {
    background: #F0FDF4;
    border-left: 4px solid #16A34A;
    border-radius: 0 8px 8px 0;
    padding: 0.45rem 0.9rem;
    margin-bottom: 0.8rem;
    font-weight: 700;
    color: #14532D;
    font-size: 1.05rem;
}

/* ---- Action recommendation callout ---- */
.action-callout {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 5px solid #16A34A;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0 0.8rem;
}
.action-callout-title {
    font-weight: 700;
    color: #14532D;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}
.action-callout-body {
    color: #166534;
    font-size: 0.93rem;
    line-height: 1.5;
}

/* ---- Intervention tier badges ---- */
.tier-critical {
    display: inline-block;
    background: #FEE2E2;
    color: #991B1B;
    font-weight: 700;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}
.tier-monitor {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    font-weight: 700;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}
.tier-routine {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    font-weight: 700;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}

/* ---- Friendly placeholder for missing artifacts ---- */
.artifact-pending {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #94A3B8;
    font-size: 0.88rem;
    text-align: center;
}

/* ---- Navigation cards on Home ---- */
.nav-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    height: 100%;
}
.nav-card-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.nav-card-title {
    font-weight: 700;
    color: #0F172A;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.nav-card-desc {
    color: #64748B;
    font-size: 0.82rem;
    line-height: 1.45;
}
/* Sidebar card */
.sidebar-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  padding: 0.75rem 0.85rem;
  margin-top: 0.75rem;
}
</style>
"""


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=title,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(markdown_text: str) -> None:
    st.markdown(
        f'<div class="dashboard-card">{markdown_text}</div>',
        unsafe_allow_html=True,
    )


def render_tab_intro(title: str, subtitle: str, color: str = "blue") -> None:
    render_section_header(title, color=color)
    st.caption(subtitle)


def open_section_card(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"**{title}**")
    if subtitle:
        st.caption(subtitle)


def close_section_card() -> None:
    # No-op helper for backward compatibility.
    return None


def show_image_card(path: Path, title: str, note: str | None = None) -> None:
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.markdown(
            '<div class="artifact-pending">Chart will appear here once the analysis pipeline has been run. Run: <code>python -m src.run_pipeline</code></div>',
            unsafe_allow_html=True,
        )
    if note:
        st.caption(note)


def pretty_name(feature: str) -> str:
    return FRIENDLY_LABELS.get(feature, feature.replace("_", " ").title())


def pretty_shap_feature(feature: str) -> str:
    if feature.startswith("num__"):
        base = feature.replace("num__", "", 1)
        return pretty_name(base)
    if feature.startswith("cat__"):
        base = feature.replace("cat__", "", 1)
        if "_" in base:
            col, value = base.split("_", 1)
            return f"{pretty_name(col)} = {value}"
        return pretty_name(base)
    return pretty_name(feature)


def format_model_table(df):
    wanted_cols = [
        "model",
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
        "cv_roc_auc_mean",
        "cv_pr_auc_mean",
    ]
    keep = [c for c in wanted_cols if c in df.columns]
    if not keep:
        return df.copy()
    out = df[keep].copy()
    sort_candidates = [
        "cv_pr_auc_mean",
        "cv_f1_mean",
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "cv_precision_mean",
        "cv_accuracy_mean",
    ]
    sort_col = next((c for c in sort_candidates if c in out.columns), None)
    if sort_col:
        out = out.sort_values(sort_col, ascending=False)
    for col in keep:
        if col != "model":
            out[col] = out[col].round(3)
    # Apply business display names to model column
    out["model"] = out["model"].map(
        lambda x: MODEL_DISPLAY_NAMES.get(x, x.replace("_", " ").title())
    )
    # Rename columns to clean business labels
    out = out.rename(columns=COMPARISON_COL_LABELS)
    return out


def style_best_row(df, best_model: str):
    # best_model should already be the mapped display name
    algo_col = COMPARISON_COL_LABELS.get("model", "Algorithm")
    best_raw = str(best_model)
    best_mapped = MODEL_DISPLAY_NAMES.get(best_raw, best_raw.replace("_", " ").title())
    best_options = {best_raw, best_mapped}

    def _highlight(row):
        col = algo_col if algo_col in row.index else "model"
        row_val = str(row.get(col, ""))
        if row_val in best_options:
            return ["background-color: #ECFDF5; font-weight: 700; color: #065F46"] * len(row)
        return [""] * len(row)

    return df.style.apply(_highlight, axis=1)


def get_risk_band(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "Low risk", "risk-low"
    if prob < 0.60:
        return "Moderate risk", "risk-medium"
    return "High risk", "risk-high"


def get_driver_bullets(features: list[str]) -> list[str]:
    lower = [f.lower() for f in features]
    bullets: list[str] = []
    if any(any(k in f for k in ["click", "active", "engagement"]) for f in lower):
        bullets.append(
            "Early engagement volume is a major risk signal; lower activity is often linked to higher withdrawal risk."
        )
    if any(any(k in f for k in ["submission", "score", "assessment", "late"]) for f in lower):
        bullets.append(
            "Assessment behavior matters: lower performance or weaker submission patterns increase risk flags."
        )
    if any(any(k in f for k in ["module", "education", "imd", "age"]) for f in lower):
        bullets.append(
            "Background and module context also influence risk, so support strategies should be tailored by student segment."
        )
    if len(bullets) < 3:
        bullets.append(
            "No single factor determines the outcome; risk reflects the combined pattern of engagement and academic signals."
        )
    return bullets[:3]


def render_sidebar_status(
    model, schema, comparison_df, page: str = "",
    *, profile: dict | None = None, metadata: dict | None = None,
    holdout_df=None,
) -> None:
    _ = page
    ready = bool(schema) and model is not None and not comparison_df.empty

    # ── Pipeline status ──
    if ready:
        st.sidebar.success("Pipeline: ready")
    else:
        st.sidebar.warning("Run the pipeline to populate results")

    # ── Dataset overview ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset")
    st.sidebar.markdown(
        "[OULAD — Open University Learning Analytics](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset)"
    )
    prof = profile or {}
    n_rows = prof.get("n_rows", 0)
    n_cols = prof.get("n_columns", 0)
    w_rate = prof.get("withdraw_rate", 0)
    if n_rows:
        st.sidebar.caption(
            f"**{n_rows:,}** enrolments, **{n_cols}** features  \n"
            f"Withdrawal rate: **{w_rate:.1%}**  \n"
            "Observation window: first 21 days"
        )

    # ── Selected model ──
    meta = metadata or {}
    best_name = meta.get("best_model_name", "")
    if best_name:
        from app_utils.constants import MODEL_DISPLAY_NAMES
        display_name = MODEL_DISPLAY_NAMES.get(best_name, best_name.replace("_", " ").title())
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Selected model")
        st.sidebar.caption(f"**{display_name}**")

        cv_pr = meta.get("best_metric_value")
        ho_pr = meta.get("holdout_pr_auc")
        ho_rec = meta.get("holdout_recall")
        ho_f1 = meta.get("holdout_f1")
        lines = []
        if cv_pr is not None:
            lines.append(f"CV PR-AUC: **{cv_pr:.3f}**")
        if ho_pr is not None:
            lines.append(f"Holdout PR-AUC: **{ho_pr:.3f}**")
        if ho_rec is not None:
            lines.append(f"Holdout Recall: **{ho_rec:.0%}**")
        if ho_f1 is not None:
            lines.append(f"Holdout F1: **{ho_f1:.3f}**")
        if lines:
            st.sidebar.caption("  \n".join(lines))

    # ── Models trained ──
    if not comparison_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Models trained")
        from app_utils.constants import MODEL_DISPLAY_NAMES
        names = [
            MODEL_DISPLAY_NAMES.get(str(m), str(m).replace("_", " ").title())
            for m in comparison_df["model"]
        ]
        st.sidebar.caption("  \n".join(f"- {n}" for n in names))


def render_header(profile: dict, metadata: dict) -> None:
    st.title("OULAD Early Withdrawal Analytics")
    n = profile.get("n_rows", 0)
    rate = profile.get("withdraw_rate", 0)
    st.caption(
        f"{n:,} students · 7 modules · 21-day window · {rate:.1%} withdrawal rate"
    )


def render_hero(profile: dict, metadata: dict) -> None:
    """Full-width gradient banner for optional landing views."""
    best_raw = str(metadata.get("best_model_name", "N/A"))
    best_display = MODEL_DISPLAY_NAMES.get(best_raw, best_raw.replace("_", " ").title())
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Student Early Warning Dashboard</div>
            <div class="hero-subtitle">
                Identifying withdrawal risk in the first 21 days of study.
                Supporting proactive student outreach across Open University modules.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Students analysed", f"{profile.get('n_rows', 0):,}")
    with c2:
        render_metric_card("Predictive signals", str(max(0, profile.get("n_columns", 1) - 1)))
    with c3:
        render_metric_card("Overall withdrawal rate", f"{profile.get('withdraw_rate', 0):.1%}")
    with c4:
        render_metric_card("Predictive model", best_display)


def render_section_header(text: str, color: str = "blue") -> None:
    """Colored left-border section header strip."""
    css_map = {
        "blue":  "section-header-blue",
        "amber": "section-header-amber",
        "green": "section-header-green",
    }
    css_class = css_map.get(color, "section-header-blue")
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_action_callout(title: str, body: str) -> None:
    """Green action recommendation box."""
    st.markdown(
        f'<div class="action-callout">'
        f'<div class="action-callout-title">{title}</div>'
        f'<div class="action-callout-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
