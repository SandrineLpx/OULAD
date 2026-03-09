"""
feature_engineering.py
=======================
Derives intervention-time-safe features from the clean 21-day modeling table.

All inputs must already be 21d-windowed (no whole-course leakage).
Features removed vs previous version:
  - activity_type_diversity_gap  (used unique_activity_types / unique_sites_visited -- whole course)
  - score_per_click_21d           (used weighted_score_sum -- redundant with weighted_avg_score)
  - assessment_density_21d via submitted_assessments (now uses submission_rate directly)

New features added:
  - Engagement trajectory acceleration (7->14d, 14->21d)
  - Zero-engagement binary flags (high signal: 38% withdrawn vs 5% retained have 0 clicks)
  - Module-relative engagement z-scores
  - Score quality per active day
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create intervention-time-safe engineered features from 21d-windowed inputs."""
    out = df.copy()

    # ------------------------------------------------------------------
    # Engagement trajectory -- captures acceleration/deceleration pattern.
    # A student with 50 clicks in week 1 and 150 in week 3 is fundamentally
    # different from one who starts high and tapers off.
    # ------------------------------------------------------------------
    out["engagement_accel_7to14"] = (
        out["first_14d_clicks"] / (out["first_7d_clicks"] + 1.0)
    )
    out["engagement_accel_14to21"] = (
        out["first_21d_clicks"] / (out["first_14d_clicks"] + 1.0)
    )
    out["active_day_growth_7to14"] = (
        out["first_14d_active_days"] / (out["first_7d_active_days"] + 1.0)
    )
    out["active_day_growth_14to21"] = (
        out["first_21d_active_days"] / (out["first_14d_active_days"] + 1.0)
    )

    # ------------------------------------------------------------------
    # Zero-engagement binary flags.
    # 37.9% of withdrawn students have zero clicks in first 21d vs 5.1% retained.
    # ------------------------------------------------------------------
    out["zero_engagement_21d"] = (out["first_21d_clicks"] == 0).astype(int)
    out["zero_engagement_7d"]  = (out["first_7d_clicks"] == 0).astype(int)
    out["zero_assessment"]     = (out["submission_rate"] == 0).astype(int)

    # ------------------------------------------------------------------
    # Engagement intensity and quality ratios (21d-windowed inputs only).
    # ------------------------------------------------------------------
    out["engagement_intensity_21d"] = (
        out["first_21d_clicks"] / (out["first_21d_active_days"] + 1.0)
    )
    out["score_per_active_day"] = (
        out["weighted_avg_score"] / (out["first_21d_active_days"] + 1.0)
    )
    out["assessment_density_21d"] = (
        out["submission_rate"] / (out["first_21d_active_days"] + 1.0)
    )
    out["late_submission_rate_safe"] = (
        out["late_submission_count"] / (out["submission_rate"].replace(0, 1.0))
    )

    # ------------------------------------------------------------------
    # Module-relative engagement z-scores.
    # A student with 50 clicks is average in high-withdrawal CCC (44.5%)
    # but outstanding in low-withdrawal GGG (11.5%).
    # ------------------------------------------------------------------
    for col in ["first_21d_clicks", "first_21d_active_days", "weighted_avg_score"]:
        if col in out.columns:
            m = out.groupby("code_module")[col].transform("mean")
            s = out.groupby("code_module")[col].transform("std").replace(0, 1.0)
            out[f"{col}_module_z"] = (out[col] - m) / s

    # ------------------------------------------------------------------
    # Socioeconomic and registration flags.
    # ------------------------------------------------------------------
    out["is_low_imd"] = out["imd_band"].fillna("missing").isin(
        ["0-10%", "10-20%", "20-30%"]
    ).astype(int)
    out["registration_gap_abs"] = np.abs(out["date_registration"].fillna(0.0))

    # ------------------------------------------------------------------
    # Semester intake flag (stable across years; B = February, J = October).
    # code_presentation is dropped before training (it encodes the year),
    # but the semester component is a structurally stable cohort signal.
    # ------------------------------------------------------------------
    if "code_presentation" in out.columns:
        out["is_feb_start"] = out["code_presentation"].str.endswith("B").astype(int)

    return out


def get_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Return named subsets of columns for ablation / feature-set experiments."""
    demographics = [
        c for c in columns if c in {
            "code_module",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "date_registration",
            "registered_before_start",
            "module_presentation_length",
            "is_low_imd",
            "registration_gap_abs",
            "is_feb_start",
        }
    ]

    engagement = [
        c for c in columns
        if (
            "click" in c
            or "active_day" in c
            or "sites" in c
            or "site" in c
            or c in {
                "pre_start_active_days",
                "first_7d_active_days",
                "first_14d_active_days",
                "first_21d_active_days",
                "engagement_intensity_21d",
                "engagement_accel_7to14",
                "engagement_accel_14to21",
                "active_day_growth_7to14",
                "active_day_growth_14to21",
                "zero_engagement_21d",
                "zero_engagement_7d",
                "min_activity_day",
                "max_activity_day",
                "first_21d_clicks_module_z",
                "first_21d_active_days_module_z",
            }
        )
    ]

    assessment = [
        c for c in columns
        if (
            "assessment" in c
            or "score" in c
            or "submission" in c
            or "delay" in c
            or "tma_" in c
            or "cma_" in c
            or "banked" in c
            or "late_" in c
            or c in {
                "zero_assessment",
                "weighted_avg_score_module_z",
                "score_per_active_day",
            }
        )
    ]

    return {
        "full": list(columns),
        "demographics_only": sorted(set(demographics)),
        "engagement_only": sorted(set(engagement)),
        "assessment_only": sorted(set(assessment)),
        "demographics_engagement": sorted(set(demographics + engagement)),
        "engagement_assessment": sorted(set(engagement + assessment)),
    }
