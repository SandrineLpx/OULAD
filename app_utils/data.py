from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, MODELS_DIR


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def load_common_artifacts() -> dict[str, Any]:
    metadata       = load_json(ARTIFACTS_DIR / "best_model_metadata.json")
    profile        = load_json(ARTIFACTS_DIR / "dataset_profile.json")
    eval_report    = load_json(ARTIFACTS_DIR / "evaluation_report.json")
    schema         = load_json(ARTIFACTS_DIR / "feature_schema.json")
    threshold_info = load_json(ARTIFACTS_DIR / "threshold_analysis.json")
    calibration    = load_json(ARTIFACTS_DIR / "calibration_metrics.json")
    temporal       = load_json(ARTIFACTS_DIR / "temporal_holdout_metrics.json")
    nn_results     = load_json(ARTIFACTS_DIR / "nn_holdout_metrics.json")

    comparison_df  = load_csv(ARTIFACTS_DIR / "model_comparison.csv")
    holdout_df     = load_csv(ARTIFACTS_DIR / "holdout_metrics.csv")
    subset_df      = load_csv(ARTIFACTS_DIR / "feature_subset_performance.csv")
    module_perf_df = load_csv(ARTIFACTS_DIR / "module_stratified_performance.csv")
    subgroup_df    = load_csv(ARTIFACTS_DIR / "demographic_subgroup_report.csv")

    model          = load_model(MODELS_DIR / "best_model.joblib")

    return {
        "metadata": metadata,
        "profile": profile,
        "eval_report": eval_report,
        "schema": schema,
        "threshold_info": threshold_info,
        "calibration": calibration,
        "temporal": temporal,
        "nn_results": nn_results,
        "comparison_df": comparison_df,
        "holdout_df": holdout_df,
        "subset_df": subset_df,
        "module_perf_df": module_perf_df,
        "subgroup_df": subgroup_df,
        "model": model,
    }
