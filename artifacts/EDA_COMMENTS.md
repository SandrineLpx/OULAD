# EDA Insights — Clean 21-Day Dataset

1. Target balance: 31.2% withdrawn vs 68.8% retained. Moderate imbalance — class weighting and PR-AUC are preferred over accuracy.
2. Remaining missingness is concentrated in assessment columns: students with no 21-day assessments receive zeros (informative absence).
3. Module CCC has 4x the withdrawal rate of GGG (44.5% vs 11.5%). Module is a strong contextual predictor — support strategies should be module-specific.
4. Retained students have a median of 149 clicks vs 28 for withdrawn — a clear separation that validates using early engagement as a predictor.
5. Withdrawn students cluster at score=0 (no submissions), while retained show a broad distribution above 60. Zero-assessment flag is a key high-signal feature.
6. Students without prior HE qualifications show higher withdrawal rates — targeted onboarding support for this group is evidence-based.
7. Engagement and assessment signals provide complementary information. The new zero_engagement_21d binary flag shows one of the strongest target correlations.
8. The engagement gap between retained and withdrawn students widens each week. Week 1-3 is the most critical intervention window.
9. Zero-click students are overwhelmingly withdrawn in every module. GGG and EEE have <5% zero-click retained students, making this flag near-deterministic there.
10. The distributional separation between classes is present as early as week 1 and grows through week 3 — confirming the 21-day window is data-rich.
11. Withdrawal rates increased from ~28% (2013) to ~34% (2014). Temporal holdout (train 2013, test 2014) is essential to detect model drift.
12. Withdrawal risk varies across all six demographic dimensions. IMD band (socioeconomic deprivation) and prior education show the sharpest gradients: low-income and less-qualified students withdraw at above-average rates. Students retaking a module (1+ previous attempts) also show elevated risk.
13. Retained students show higher engagement acceleration in the final week. Students whose weekly clicks plateau or drop (ratio ≤ 1) are higher risk.
14. Withdrawn students cluster near zero log-clicks in the first 21 days. Most at-risk students never meaningfully engage with the platform.
15. Clicks, active days, and score all separate the two groups. 74% of withdrawn students had a score of 0 at day 21, vs 54% of retained.
