from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COL = "withdrawn_flag"
STUDENT_ID_COL = "id_student"

# ---------------------------------------------------------------------------
# Columns to NEVER use as model inputs
# ---------------------------------------------------------------------------

# Row identifiers
ID_COLS = ["id_student", "code_presentation"]

# Whole-course leakage: aggregated over the entire course, not just first 21 days.
# Confirmed: sum(clicks_activity_*) == total_clicks (whole-course), NOT first_21d_clicks.
LEAKY_COLS = [
    "has_unregistered_flag",           # direct withdrawal proxy (target leak)
    "total_clicks",                    # whole course
    "student_site_day_events",         # whole course
    "active_days",                     # whole course
    "unique_sites_visited",            # whole course
    "avg_clicks_per_student_site_day", # whole course
    "max_clicks_single_event",         # whole course
    "raw_log_rows",                    # whole course
    "avg_clicks_per_active_day",       # whole course
    "site_diversity_ratio",            # whole course
    "pre_start_click_share",           # = pre_start / total_clicks -> future denominator
    "first_21d_click_share",           # = first_21d / total_clicks -> future denominator
    "unique_activity_types",           # whole course
    # 18 per-activity-type click columns -- their sum equals total_clicks (whole course)
    "clicks_activity_dataplus",
    "clicks_activity_dualpane",
    "clicks_activity_externalquiz",
    "clicks_activity_forumng",
    "clicks_activity_glossary",
    "clicks_activity_homepage",
    "clicks_activity_htmlactivity",
    "clicks_activity_oucollaborate",
    "clicks_activity_oucontent",
    "clicks_activity_ouelluminate",
    "clicks_activity_ouwiki",
    "clicks_activity_page",
    "clicks_activity_questionnaire",
    "clicks_activity_quiz",
    "clicks_activity_resource",
    "clicks_activity_sharedsubpage",
    "clicks_activity_subpage",
    "clicks_activity_url",
]

# Zero-variance or fully redundant columns (r=1.00 groups).
# Keep the single best representative from each group.
REDUNDANT_COLS = [
    "score_std",               # zero variance across all 32,593 rows
    # submission group (keep submission_rate)
    "assessment_records",
    "unique_assessments",
    "submitted_assessments",
    # score group (keep weighted_avg_score)
    "avg_score",
    "median_score",
    "min_score",
    "max_score",
    # weight group (keep weighted_avg_score)
    "avg_weight",
    "total_weight",
    "weighted_score_sum",
    # submission timing group (keep avg_submission_delay + max_submission_delay)
    "median_submission_delay",
    "min_submission_delay",
    # near-zero sparse
    "missing_score_count",     # 99.9% zeros
    "banked_rate",             # 99.1% zeros
    # TMA/CMA redundant weighted sums
    "tma_weighted_score_sum",
    "cma_weighted_score_sum",
    # date span (keep first_assessment_day)
    "last_assessment_day",
    # Near-constant: 99.1% of rows share the same value -- no meaningful signal
    "registered_before_start",  # 99.1% = 1; withdrawal rate for 0 vs 1 differs by <0.4pp
    "banked_count",             # 99.1% = 0; only 279 non-zero rows
]

DROP_COLS = ID_COLS + LEAKY_COLS + REDUNDANT_COLS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Clean dataset (rebuilt from processed 21d files -- no leakage)
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "oulad_modeling_table_clean.csv"
# Legacy path (kept for reference; not used by the new pipeline)
DATA_PATH = PROJECT_ROOT / "data" / "oulad_modeling_table_21d.csv"

# Processed 21-day aggregation sources
VLE_21D_PATH   = PROJECT_ROOT / "data" / "processed" / "studentVle_aggregated_21d.csv"
ASMNT_21D_PATH = PROJECT_ROOT / "data" / "processed" / "studentAssessment_aggregated_21d.csv"

# Raw sources
RAW_INFO_PATH    = PROJECT_ROOT / "data" / "raw" / "studentInfo.csv"
RAW_REG_PATH     = PROJECT_ROOT / "data" / "raw" / "studentRegistration.csv"
RAW_COURSES_PATH = PROJECT_ROOT / "data" / "raw" / "courses.csv"

# Output directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR    = PROJECT_ROOT / "models"

for _path in [ARTIFACTS_DIR, MODELS_DIR, ARTIFACTS_DIR / "eda"]:
    _path.mkdir(parents=True, exist_ok=True)
