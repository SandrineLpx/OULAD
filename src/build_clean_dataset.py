"""
build_clean_dataset.py
======================
Constructs the leak-free 21-day modeling table from raw and processed sources.

Key design decisions
--------------------
* Base: studentInfo.csv (32,593 enrollment rows -- one per student×module×presentation)
* Target: binary withdrawn_flag = (final_result == 'Withdrawn')
* date_unregistration is dropped -- it IS the withdrawal event (target leakage)
* From VLE: only explicitly time-windowed columns (pre_start_*, first_7d_*,
  first_14d_*, first_21d_*, min/max_activity_day) are kept.
  Whole-course columns (total_clicks, clicks_activity_*, active_days, etc.) are excluded.
* From assessments: redundant r=1.00 groups and zero-variance columns are dropped.
* Students with no VLE or assessment activity in the 21d window get zeros (left join + fillna).
"""
from __future__ import annotations

import pandas as pd

from src.config import (
    ASMNT_21D_PATH,
    DATA_CLEAN_PATH,
    RAW_COURSES_PATH,
    RAW_INFO_PATH,
    RAW_REG_PATH,
    TARGET_COL,
    VLE_21D_PATH,
)

# ---------------------------------------------------------------------------
# Column white-lists
# ---------------------------------------------------------------------------

# Only explicitly 21d-windowed VLE signals are safe at prediction time.
VLE_KEEP = [
    "code_module", "code_presentation", "id_student",
    # pre-start engagement (before course day 0)
    "pre_start_clicks",
    "pre_start_active_days",
    "pre_start_sites",
    "pre_start_avg_clicks_per_student_site_day",
    # first 7-day window
    "first_7d_clicks",
    "first_7d_active_days",
    "first_7d_sites",
    "first_7d_avg_clicks_per_student_site_day",
    # first 14-day window
    "first_14d_clicks",
    "first_14d_active_days",
    "first_14d_sites",
    "first_14d_avg_clicks_per_student_site_day",
    # first 21-day window
    "first_21d_clicks",
    "first_21d_active_days",
    "first_21d_sites",
    "first_21d_avg_clicks_per_student_site_day",
    # activity day range within the 21d window
    "min_activity_day",
    "max_activity_day",
]

# Assessment features that survive redundancy checks (see config.py REDUNDANT_COLS).
ASMNT_KEEP = [
    "code_module", "code_presentation", "id_student",
    # submission behaviour
    "submission_rate",
    "weighted_avg_score",
    # submission timing
    "avg_submission_delay",
    "max_submission_delay",
    # submission outcome counts
    "late_submission_count",
    "early_or_ontime_count",
    "late_submission_rate",
    "banked_count",
    # assessment calendar
    "first_assessment_day",
    # TMA (Tutor-Marked Assignments)
    "tma_count",
    "tma_submitted",
    "tma_avg_score",
    "tma_avg_delay",
    "tma_late_count",
    "tma_submission_rate",
    # CMA (Computer-Marked Assignments)
    "cma_count",
    "cma_submitted",
    "cma_avg_score",
    "cma_avg_delay",
    "cma_late_count",
    "cma_submission_rate",
]

# Numeric VLE columns (filled with 0 when student has no VLE record)
VLE_NUMERIC = [c for c in VLE_KEEP if c not in ("code_module", "code_presentation", "id_student")]

# Numeric assessment columns (filled with 0 when student has no assessment record)
ASMNT_NUMERIC = [c for c in ASMNT_KEEP if c not in ("code_module", "code_presentation", "id_student")]


def build_clean_dataset(output_path=None) -> pd.DataFrame:
    """
    Build and save the clean 21-day modeling table.

    Returns
    -------
    pd.DataFrame  The assembled dataset (32,593 rows).
    """
    out_path = output_path or DATA_CLEAN_PATH

    # ------------------------------------------------------------------
    # 1. Base: student demographics + outcome
    # ------------------------------------------------------------------
    info = pd.read_csv(RAW_INFO_PATH)
    # Binary target: 1 = Withdrawn, 0 = Pass / Fail / Distinction
    info[TARGET_COL] = (info["final_result"] == "Withdrawn").astype(int)
    info = info.drop(columns=["final_result"])

    # ------------------------------------------------------------------
    # 2. Registration timing (safe -- known at day 0)
    # ------------------------------------------------------------------
    reg = pd.read_csv(RAW_REG_PATH)
    # date_unregistration is the withdrawal event itself -- drop it (target leak)
    reg = reg.drop(columns=["date_unregistration"], errors="ignore")
    reg["registered_before_start"] = (reg["date_registration"] <= 0).astype(int)

    df = info.merge(
        reg[["code_module", "code_presentation", "id_student",
             "date_registration", "registered_before_start"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 3. Course metadata
    # ------------------------------------------------------------------
    courses = pd.read_csv(RAW_COURSES_PATH)
    df = df.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left",
    )

    # ------------------------------------------------------------------
    # 4. 21-day VLE engagement (time-windowed only)
    # ------------------------------------------------------------------
    vle_raw = pd.read_csv(VLE_21D_PATH)
    # Keep only columns that exist in the file and are in our whitelist
    vle_cols = [c for c in VLE_KEEP if c in vle_raw.columns]
    vle = vle_raw[vle_cols]

    df = df.merge(
        vle,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    # Students with no VLE record had zero activity in the first 21 days
    for col in VLE_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ------------------------------------------------------------------
    # 5. 21-day assessment features
    # ------------------------------------------------------------------
    asmnt_raw = pd.read_csv(ASMNT_21D_PATH)
    asmnt_cols = [c for c in ASMNT_KEEP if c in asmnt_raw.columns]
    asmnt = asmnt_raw[asmnt_cols]

    df = df.merge(
        asmnt,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    # Students with no 21d assessment had zero submissions
    for col in ASMNT_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    df.to_csv(out_path, index=False)
    print(f"Clean dataset saved: {out_path}")
    print(f"  Shape : {df.shape}")
    print(f"  Target: {df[TARGET_COL].value_counts().to_dict()}")
    print(f"  Nulls : {df.isnull().sum().sum()} total")
    return df


if __name__ == "__main__":
    build_clean_dataset()
