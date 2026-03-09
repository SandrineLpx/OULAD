from __future__ import annotations


# ---------------------------------------------------------------------------
# Model display names — maps raw CSV/joblib names to clean business labels
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "lightgbm":            "LightGBM (Gradient Boosting)",
    "gradient_boosting":   "Gradient Boosting",
    "random_forest":       "Random Forest",
    "logistic_l2":         "Logistic Regression (L2)",
    "logistic_l1":         "Logistic Regression (L1)",
    "logistic_regression": "Logistic Regression",
    "ridge":               "Logistic Regression (L2 Ridge)",
    "lasso":               "Logistic Regression (L1 Lasso)",
    "decision_tree":       "Decision Tree",
    "svm":                 "Support Vector Machine",
    "baseline_dummy":      "Baseline (Always-Majority)",
    "nn":                  "Neural Network (MLP)",
    "neural_network_keras": "Neural Network (MLP)",
    # best_model.joblib is loaded as "best_model" by load_available_models()
    "best_model":          "Gradient Boosting (selected)",
}

# ---------------------------------------------------------------------------
# Model comparison table — column header labels for st.dataframe
# ---------------------------------------------------------------------------
COMPARISON_COL_LABELS: dict[str, str] = {
    "model":             "Algorithm",
    "cv_accuracy_mean":  "Accuracy",
    "cv_precision_mean": "Precision",
    "cv_recall_mean":    "Recall",
    "cv_f1_mean":        "F1 Score",
    "cv_roc_auc_mean":   "ROC AUC",
    "cv_pr_auc_mean":    "PR AUC \u25b2",
}

# ---------------------------------------------------------------------------
# Friendly feature labels for display
# ---------------------------------------------------------------------------
FRIENDLY_LABELS: dict[str, str] = {
    # Demographics
    "code_module": "Module",
    "gender": "Gender",
    "region": "Region",
    "highest_education": "Highest Education",
    "imd_band": "Socioeconomic background (IMD band)",
    "age_band": "Age Band",
    "num_of_prev_attempts": "Prior attempts at this module",
    "studied_credits": "Studied Credits",
    "disability": "Disability",
    "date_registration": "Registration Day (relative to course start)",
    "registered_before_start": "Registered Before Course Start",
    "module_presentation_length": "Course Length (days)",
    # VLE engagement
    "pre_start_clicks": "Pre-Start Clicks",
    "pre_start_active_days": "Pre-Start Active Days",
    "pre_start_sites": "Pre-Start Sites Visited",
    "first_7d_clicks": "Clicks in First 7 Days",
    "first_7d_active_days": "Active Days in First 7 Days",
    "first_7d_sites": "Sites in First 7 Days",
    "first_14d_clicks": "Clicks in First 14 Days",
    "first_14d_active_days": "Active Days in First 14 Days",
    "first_14d_sites": "Sites in First 14 Days",
    "first_21d_clicks": "Clicks in First 21 Days",
    "first_21d_active_days": "Active Days in First 21 Days",
    "first_21d_sites": "Sites in First 21 Days",
    "first_21d_avg_clicks_per_student_site_day": "Avg Clicks / Site-Day (21d)",
    "min_activity_day": "Earliest Active Day",
    "max_activity_day": "Latest Active Day (within 21d)",
    # Assessment
    "submission_rate": "Submission Rate",
    "weighted_avg_score": "Weighted Average Score",
    "avg_submission_delay": "Avg Submission Delay (days)",
    "max_submission_delay": "Max Submission Delay (days)",
    "late_submission_count": "Late Submissions",
    "early_or_ontime_count": "On-Time or Early Submissions",
    "late_submission_rate": "Late submission rate (first 21 days)",
    "banked_count": "Banked Assessments",
    "first_assessment_day": "First Assessment Day",
    "tma_count": "TMA Count",
    "tma_submitted": "TMAs Submitted",
    "tma_avg_score": "TMA Average Score",
    "tma_avg_delay": "TMA Avg Delay",
    "tma_late_count": "TMA Late Submissions",
    "tma_submission_rate": "TMA Submission Rate",
    "cma_count": "CMA Count",
    "cma_submitted": "CMAs Submitted",
    "cma_avg_score": "CMA Average Score",
    "cma_avg_delay": "CMA Avg Delay",
    "cma_late_count": "CMA Late Submissions",
    "cma_submission_rate": "CMA Submission Rate",
    # Engineered features
    "engagement_accel_7to14": "Engagement Acceleration (7\u219214d)",
    "engagement_accel_14to21": "Engagement Acceleration (14\u219221d)",
    "active_day_growth_7to14": "Active Day Growth (7\u219214d)",
    "active_day_growth_14to21": "Active Day Growth (14\u219221d)",
    "zero_engagement_21d": "Zero Engagement Flag (21d)",
    "zero_engagement_7d": "Zero Engagement Flag (7d)",
    "zero_assessment": "Zero Assessment Flag",
    "engagement_intensity_21d": "Average daily engagement (first 21 days)",
    "score_per_active_day": "Score per Active Day",
    "assessment_density_21d": "Assessment Density (21d)",
    "late_submission_rate_safe": "Late Submission Rate (smoothed)",
    "is_low_imd": "Low IMD Band (bottom 30%)",
    "registration_gap_abs": "Registration Gap (absolute days)",
    "is_feb_start": "February Intake (vs October)",
    "first_21d_clicks_module_z": "Clicks vs Module Average (z-score)",
    "first_21d_active_days_module_z": "Active Days vs Module Average (z-score)",
    "weighted_avg_score_module_z": "Score vs Module Average (z-score)",
}

# ---------------------------------------------------------------------------
# EDA chart captions — business-friendly language throughout
# ---------------------------------------------------------------------------
EDA_NOTES: dict[str, str] = {
    "01_target_distribution.png": (
        "The target is moderately imbalanced: 31.2% of students withdrew vs 68.8% retained. "
        "This imbalance means accuracy alone is misleading — a model that always predicts 'retained' "
        "would be 69% accurate but catch zero at-risk students. "
        "To address this, all models use class_weight='balanced' and selection uses PR-AUC "
        "(precision-recall area under curve), which penalises false positives on the minority class."
    ),
    "03_withdrawal_by_module.png": (
        "Module CCC has a withdrawal rate 4x higher than GGG (44.5% vs 11.5%). "
        "This large variation across modules suggests that module context is an important predictor. "
        "Module-specific support strategies and module-relative feature z-scores help capture this signal."
    ),
    "04_engagement_by_outcome.png": (
        "Students who stay engaged click substantially more in the first 21 days. "
        "Retained students have a median of 149 clicks vs 28 for withdrawn — a 5x difference. "
        "Early platform activity is one of the strongest early warning signals available."
    ),
    "05_assessment_by_outcome.png": (
        "Students who withdraw cluster at a score of zero on early assessments. "
        "Missing or failing the first assessment is a high-confidence risk indicator. "
        "The zero-assessment binary flag was engineered as a feature because of this strong signal."
    ),
    "06_withdrawal_by_education.png": (
        "Students with lower prior qualifications withdraw at higher rates. "
        "Those without prior HE qualifications show the largest gap relative to the 31% average. "
        "Onboarding and foundational support can be targeted to this group."
    ),
    "07_correlation_heatmap.png": (
        "Strongest correlation with withdrawal: zero_engagement_21d (r = 0.42) — students who never log in "
        "are overwhelmingly at risk. Active days (r = -0.32) and clicks (r = -0.22) confirm that consistent "
        "platform use is protective. Assessment and engagement features cluster separately, "
        "both carrying independent predictive signal. This justifies including both feature families in the model."
    ),
    "08_engagement_trend.png": (
        "The gap between retained and withdrawn students widens every week in the first three weeks. "
        "By day 21, the separation is substantial, making early outreach the highest-leverage intervention point. "
        "This widening trend supports the choice of a 21-day observation window."
    ),
    "09_zero_engagement_heatmap.png": (
        "Students with zero clicks in the first 21 days almost always withdraw, "
        "and this pattern holds across every module. "
        "A single binary flag for zero activity is one of the most discriminating signals in the model, "
        "motivating its inclusion as an engineered feature."
    ),
    "10_engagement_distributions.png": (
        "The difference in engagement levels between retained and withdrawn groups is visible from week 1 "
        "and grows through week 3. Even the week-1 window shows meaningful separation. "
        "This confirms the 21-day observation window provides rich early warning information."
    ),
    "11_temporal_shift.png": (
        "Withdrawal rates rose by approximately 6 percentage points from 2013 to 2014. "
        "This cohort shift means a model trained on 2013 data must be tested on 2014 "
        "to confirm it generalises — hence the temporal holdout evaluation strategy."
    ),
    "12_withdrawal_by_demographics.png": (
        "Withdrawal risk varies across all six demographic dimensions. "
        "IMD band (socioeconomic deprivation) and prior education show the sharpest gradients — "
        "low-income and less-qualified students withdraw at above-average rates. "
        "Students retaking a module (1+ previous attempts) also show elevated risk. "
        "Dashed line = 31% overall withdrawal rate."
    ),
    "13_engagement_acceleration.png": (
        "Retained students show increasing click activity from week 2 to week 3, "
        "while withdrawn students plateau or decline. "
        "A declining engagement trajectory during this window is a meaningful risk signal, "
        "captured by the engagement acceleration features in the model."
    ),
    "14_clicks_by_outcome.png": (
        "Withdrawn students (red) cluster near zero platform activity in the first 21 days, "
        "while retained students (blue) spread across a wide range of usage. "
        "The density overlap is minimal at the low end, confirming that very low engagement "
        "is a near-certain indicator of withdrawal risk."
    ),
    "15_boxplots_by_outcome.png": (
        "All three signals separate the groups clearly. Clicks and active days show large median differences, "
        "and the distributions have limited overlap. "
        "74% of withdrawn students had a score of zero at day 21 (no submissions), "
        "vs 54% of retained — making zero assessment a strong individual risk flag."
    ),
}

# ---------------------------------------------------------------------------
# Model performance chart captions — plain language, no jargon
# ---------------------------------------------------------------------------
MODEL_NOTES: dict[str, str] = {
    "model_comparison.png": (
        "Each bar shows how reliably the algorithm detects withdrawals on unseen student data. "
        "LightGBM leads the comparison and was selected as the production model."
    ),
    "roc_curve.png": (
        "Overall ability to separate students who withdraw from those who stay. "
        "A score closer to 1.0 indicates stronger discrimination."
    ),
    "pr_curve.png": (
        "Shows the trade-off between catching more at-risk students (recall) "
        "and keeping false alarms low (precision) across different sensitivity settings."
    ),
    "confusion_matrix.png": (
        "Counts of correct flags and errors at the standard sensitivity setting. "
        "Rows = actual outcome; columns = model prediction."
    ),
    "feature_importance.png": (
        "The signals the model relies on most when scoring a student. "
        "Higher bars = stronger influence on the withdrawal risk score."
    ),
    "threshold_curve.png": (
        "How the model's flag rate and accuracy change with the sensitivity setting. "
        "A lower threshold flags more students; a higher threshold flags fewer with greater confidence."
    ),
    "calibration_curve.png": (
        "Checks whether the model's probability scores are trustworthy. "
        "A well-calibrated model means a 70% risk score genuinely reflects ~70% likelihood of withdrawal."
    ),
    "temporal_holdout_comparison.png": (
        "Compares performance on the training cohort versus the following year's cohort. "
        "Stable scores confirm the model generalises to new student groups."
    ),
}

# ---------------------------------------------------------------------------
# Key insights — business-friendly bullets, no data-science jargon
# ---------------------------------------------------------------------------
INSIGHT_DEFAULTS: list[str] = [
    "Nearly 1 in 3 students (31%) withdraws before completing their module — the model focuses on flagging these students early.",
    "Withdrawal risk varies sharply by module: some courses see rates 4\u00d7 higher than others, pointing to targeted support opportunities.",
    "Students who show zero platform activity in the first 21 days withdraw at dramatically higher rates than active peers.",
    "The engagement gap between students who stay and those who leave widens every week in the first three weeks — making early outreach the highest-leverage intervention.",
    "Weaker early assessment scores and late or missed submissions are independent warning signals on top of low engagement.",
    "Background factors (prior education, socioeconomic band) influence risk and allow support strategies to be tailored by student segment.",
    "The model uses only information available in the first 21 days, so intervention can begin before disengagement is entrenched.",
]
