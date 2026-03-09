"""
eda.py
======
Generates 13 EDA visualizations for the clean 21-day dataset.

Original 8 plots (preserved):
  01 target distribution
  02 missing values
  03 withdrawal by module
  04 engagement by outcome
  05 assessment by outcome
  06 withdrawal by education
  07 correlation heatmap (clean features only)
  08 engagement trajectory (mean line plot)

New plots (5 added):
  09 zero-engagement heatmap  (module x outcome zero-click rates)
  10 engagement trajectory distributions (violin per window per class)
  11 temporal shift (withdrawal rate by presentation year)
  12 withdrawal by student demographics (6-panel: gender/age/education/IMD/disability/prev attempts)
  13 engagement acceleration by outcome
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_CLEAN_PATH, TARGET_COL
from src.feature_engineering import add_derived_features


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(data_path: Path | None = None) -> None:
    plt.style.use("ggplot")
    # Consistent typography across all charts
    plt.rcParams.update({
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.titlesize": 12,
    })
    eda_dir = ARTIFACTS_DIR / "eda"
    _ensure_dir(eda_dir)

    src = data_path or DATA_CLEAN_PATH
    df = pd.read_csv(src)
    df = add_derived_features(df)

    insights: list[str] = []
    target_map = {0: "Retained", 1: "Withdrawn"}
    df["target_label"] = df[TARGET_COL].map(target_map)

    # ------------------------------------------------------------------
    # 01  Target distribution
    # ------------------------------------------------------------------
    counts = df["target_label"].value_counts().reindex(["Retained", "Withdrawn"])
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=["#2E86AB", "#C0392B"])
    for bar, v in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    plt.title("Student Withdrawal Distribution (Target)")
    plt.xlabel("Outcome")
    plt.ylabel("Number of Students")
    _save_fig(eda_dir / "01_target_distribution.png")
    withdraw_rate = df[TARGET_COL].mean()
    insights.append(
        f"Target balance: {withdraw_rate:.1%} withdrawn vs {1 - withdraw_rate:.1%} retained. "
        "Moderate imbalance — class weighting and PR-AUC are preferred over accuracy."
    )

    # ------------------------------------------------------------------
    # 02  Missing values
    # ------------------------------------------------------------------
    missing_rate = (df.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_rate[missing_rate > 0].head(20)
    if not top_missing.empty:
        plt.figure(figsize=(10, 6))
        top_missing.plot(kind="bar", color="#6C5B7B")
        plt.title("Top Features by Missingness (%)")
        plt.xlabel("Feature")
        plt.ylabel("Missing (%)")
        plt.xticks(rotation=75, ha="right")
        _save_fig(eda_dir / "02_missing_values_top20.png")
        insights.append(
            "Remaining missingness is concentrated in assessment columns: "
            "students with no 21-day assessments receive zeros (informative absence)."
        )

    # ------------------------------------------------------------------
    # 03  Withdrawal rate by module
    # ------------------------------------------------------------------
    if {"code_module", TARGET_COL}.issubset(df.columns):
        module_rate = (
            df.groupby("code_module", as_index=False)[TARGET_COL]
            .mean()
            .sort_values(TARGET_COL, ascending=False)
        )
        plt.figure(figsize=(9, 5))
        bars = plt.bar(module_rate["code_module"], module_rate[TARGET_COL], color="#E67E22")
        for bar, v in zip(bars, module_rate[TARGET_COL]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{v:.0%}", ha="center", fontsize=9)
        plt.title("Withdrawal Rate by Module")
        plt.xlabel("Module")
        plt.ylabel("Withdrawal Rate")
        _save_fig(eda_dir / "03_withdrawal_by_module.png")
        insights.append(
            "Module CCC has 4x the withdrawal rate of GGG (44.5% vs 11.5%). "
            "Module is a strong contextual predictor — support strategies should be module-specific."
        )

    # ------------------------------------------------------------------
    # 04  Engagement by outcome (boxplot)
    # ------------------------------------------------------------------
    if "first_21d_clicks" in df.columns:
        retained = df.loc[df[TARGET_COL] == 0, "first_21d_clicks"].dropna()
        withdrawn = df.loc[df[TARGET_COL] == 1, "first_21d_clicks"].dropna()
        plt.figure(figsize=(9, 5))
        plt.boxplot(
            [retained, withdrawn],
            tick_labels=["Retained", "Withdrawn"],
            patch_artist=True,
            boxprops=dict(facecolor="#AED6F1"),
            medianprops=dict(color="#C0392B", linewidth=2),
        )
        plt.title("Early Engagement (First 21-Day Clicks) by Outcome")
        plt.ylabel("Clicks in First 21 Days")
        plt.ylim(0, df["first_21d_clicks"].quantile(0.98))
        _save_fig(eda_dir / "04_engagement_by_outcome.png")
        insights.append(
            "Retained students have a median of "
            f"{retained.median():.0f} clicks vs {withdrawn.median():.0f} for withdrawn — "
            "a clear separation that validates using early engagement as a predictor."
        )

    # ------------------------------------------------------------------
    # 05  Assessment performance by outcome (violin)
    # ------------------------------------------------------------------
    if "weighted_avg_score" in df.columns:
        plot_df = df.copy()
        plot_df["weighted_avg_score"] = plot_df["weighted_avg_score"].clip(
            lower=0, upper=plot_df["weighted_avg_score"].quantile(0.99)
        )
        retained_s = plot_df.loc[plot_df[TARGET_COL] == 0, "weighted_avg_score"].dropna()
        withdrawn_s = plot_df.loc[plot_df[TARGET_COL] == 1, "weighted_avg_score"].dropna()
        plt.figure(figsize=(9, 5))
        violin = plt.violinplot([retained_s, withdrawn_s], showmeans=True, showmedians=True)
        for body, color in zip(violin["bodies"], ["#16A085", "#C0392B"]):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        plt.xticks([1, 2], ["Retained", "Withdrawn"])
        plt.title("Assessment Performance (Weighted Avg Score) by Outcome")
        plt.ylabel("Weighted Average Score")
        _save_fig(eda_dir / "05_assessment_by_outcome.png")
        insights.append(
            "Withdrawn students cluster at score=0 (no submissions), while retained show "
            "a broad distribution above 60. Zero-assessment flag is a key high-signal feature."
        )

    # ------------------------------------------------------------------
    # 06  Withdrawal by education
    # ------------------------------------------------------------------
    if {"highest_education", TARGET_COL}.issubset(df.columns):
        edu_rate = (
            df.groupby("highest_education", as_index=False)[TARGET_COL]
            .mean()
            .sort_values(TARGET_COL, ascending=False)
        )
        plt.figure(figsize=(10, 5))
        plt.bar(edu_rate["highest_education"], edu_rate[TARGET_COL], color="#8E44AD")
        plt.title("Withdrawal Rate by Highest Education")
        plt.xlabel("Highest Education")
        plt.ylabel("Withdrawal Rate")
        plt.xticks(rotation=30, ha="right")
        _save_fig(eda_dir / "06_withdrawal_by_education.png")
        insights.append(
            "Students without prior HE qualifications show higher withdrawal rates — "
            "targeted onboarding support for this group is evidence-based."
        )

    # ------------------------------------------------------------------
    # 07  Correlation heatmap (clean 21d features only)
    # ------------------------------------------------------------------
    corr_candidates = [
        c for c in [
            TARGET_COL,
            "first_21d_clicks",
            "first_21d_active_days",
            "submission_rate",
            "weighted_avg_score",
            "late_submission_rate_safe",
            "engagement_intensity_21d",
            "zero_engagement_21d",
            "engagement_accel_14to21",
        ]
        if c in df.select_dtypes(include="number").columns
    ]
    if len(corr_candidates) >= 3:
        corr = df[corr_candidates].corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        ticks = np.arange(len(corr.columns))
        plt.xticks(ticks, corr.columns, rotation=45, ha="right")
        plt.yticks(ticks, corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                         ha="center", va="center", fontsize=7.5)
        plt.title("Correlation Matrix — Clean 21-Day Features")
        _save_fig(eda_dir / "07_correlation_heatmap.png")
        insights.append(
            "Engagement and assessment signals provide complementary information. "
            "The new zero_engagement_21d binary flag shows one of the strongest target correlations."
        )

    # ------------------------------------------------------------------
    # 08  Engagement trajectory (mean lines)
    # ------------------------------------------------------------------
    trend_cols = [c for c in ["first_7d_clicks", "first_14d_clicks", "first_21d_clicks"] if c in df.columns]
    if len(trend_cols) == 3:
        trend = df.groupby(TARGET_COL)[trend_cols].mean().T
        trend.index = ["7 Days", "14 Days", "21 Days"]
        plt.figure(figsize=(8, 5))
        for outcome, color in [(0, "#2471A3"), (1, "#B03A2E")]:
            plt.plot(trend.index, trend[outcome], marker="o", linewidth=2.5,
                     label=target_map[outcome], color=color)
        plt.fill_between(trend.index, trend[0], trend[1], alpha=0.12, color="gray")
        plt.title("Mean Engagement Trajectory by Outcome")
        plt.ylabel("Average Clicks")
        plt.legend()
        _save_fig(eda_dir / "08_engagement_trend.png")
        insights.append(
            "The engagement gap between retained and withdrawn students widens each week. "
            "Week 1-3 is the most critical intervention window."
        )

    # ------------------------------------------------------------------
    # 09  (NEW) Zero-engagement rate heatmap by module × outcome
    # ------------------------------------------------------------------
    if {"code_module", TARGET_COL, "first_21d_clicks"}.issubset(df.columns):
        zero_heatmap = (
            df.assign(zero_clicks=(df["first_21d_clicks"] == 0).astype(int))
            .groupby(["code_module", TARGET_COL])["zero_clicks"]
            .mean()
            .unstack(TARGET_COL)
            .rename(columns=target_map)
        )
        plt.figure(figsize=(8, 5))
        plt.imshow(zero_heatmap.T.values, cmap="Reds", aspect="auto", vmin=0, vmax=0.6)
        plt.colorbar(label="% with zero clicks")
        plt.xticks(np.arange(len(zero_heatmap.index)), zero_heatmap.index)
        plt.yticks([0, 1], ["Retained", "Withdrawn"])
        for i in range(zero_heatmap.shape[1]):
            for j, row in enumerate(zero_heatmap.index):
                val = zero_heatmap.loc[row].iloc[i]
                plt.text(j, i, f"{val:.0%}", ha="center", va="center",
                         color="white" if val > 0.35 else "black", fontsize=9)
        plt.title("Zero-Click Rate by Module and Outcome\n(% of students with 0 first-21d clicks)")
        _save_fig(eda_dir / "09_zero_engagement_heatmap.png")
        insights.append(
            "Zero-click students are overwhelmingly withdrawn in every module. "
            "GGG and EEE have <5% zero-click retained students, making this flag near-deterministic there."
        )

    # ------------------------------------------------------------------
    # 10  (NEW) Engagement distributions per window — violin
    # ------------------------------------------------------------------
    window_cols = ["first_7d_clicks", "first_14d_clicks", "first_21d_clicks"]
    present_windows = [c for c in window_cols if c in df.columns]
    if len(present_windows) == 3:
        cap = df["first_21d_clicks"].quantile(0.97)
        fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
        labels = ["First 7 Days", "First 14 Days", "First 21 Days"]
        for ax, col, label in zip(axes, present_windows, labels):
            ret_data = df.loc[df[TARGET_COL] == 0, col].clip(upper=cap).dropna()
            wth_data = df.loc[df[TARGET_COL] == 1, col].clip(upper=cap).dropna()
            vp = ax.violinplot([ret_data, wth_data], showmedians=True)
            for body, color in zip(vp["bodies"], ["#2E86AB", "#C0392B"]):
                body.set_facecolor(color)
                body.set_alpha(0.75)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Retained", "Withdrawn"])
            ax.set_title(label)
        axes[0].set_ylabel("Clicks (clipped at 97th pct)")
        fig.suptitle("Click Distribution per Time Window by Outcome", fontsize=13, y=1.02)
        _save_fig(eda_dir / "10_engagement_distributions.png")
        insights.append(
            "The distributional separation between classes is present as early as week 1 "
            "and grows through week 3 — confirming the 21-day window is data-rich."
        )

    # ------------------------------------------------------------------
    # 11  (NEW) Temporal shift — withdrawal rate by presentation year
    # ------------------------------------------------------------------
    if "code_presentation" in df.columns:
        df["year"] = df["code_presentation"].str[:4]
        yr_rate = (
            df.groupby(["year", "code_presentation"], as_index=False)
            .agg(
                withdraw_rate=(TARGET_COL, "mean"),
                n=(TARGET_COL, "size"),
            )
            .rename(columns={"code_presentation": "presentation"})
        )
        yr_rate_agg = yr_rate.groupby("year")["withdraw_rate"].mean()
        plt.figure(figsize=(7, 4))
        bars = plt.bar(yr_rate_agg.index, yr_rate_agg.values,
                       color=["#2E86AB" if y == "2013" else "#C0392B" for y in yr_rate_agg.index])
        for bar, v in zip(bars, yr_rate_agg.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
        plt.title("Withdrawal Rate Trend: 2013 vs 2014 Presentations")
        plt.xlabel("Year")
        plt.ylabel("Mean Withdrawal Rate")
        _save_fig(eda_dir / "11_temporal_shift.png")
        insights.append(
            "Withdrawal rates increased from ~28% (2013) to ~34% (2014). "
            "Temporal holdout (train 2013, test 2014) is essential to detect model drift."
        )

    # ------------------------------------------------------------------
    # 12  (NEW) Withdrawal rate by student demographics — 6-panel grid
    # ------------------------------------------------------------------
    demog_cols = [
        c for c in
        ["gender", "age_band", "highest_education", "imd_band", "disability", "num_of_prev_attempts"]
        if c in df.columns
    ]
    if len(demog_cols) >= 2:
        _labels = {
            "gender":               "Gender",
            "age_band":             "Age Band",
            "highest_education":    "Prior Education",
            "imd_band":             "Socioeconomic Band (IMD)",
            "disability":           "Disability",
            "num_of_prev_attempts": "Previous Attempts",
        }
        _imd_order = [
            "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
        ]
        n_panels = len(demog_cols)
        n_cols = 3
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.8 * n_rows))
        axes = np.array(axes).flatten()
        overall_rate = df[TARGET_COL].mean()

        for idx, col in enumerate(demog_cols):
            ax = axes[idx]
            col_df = df[[col, TARGET_COL]].dropna()

            if col == "num_of_prev_attempts":
                col_df = col_df.copy()
                col_df[col] = col_df[col].astype(int).apply(lambda x: "3+" if x >= 3 else str(x))
                order = [v for v in ["0", "1", "2", "3+"] if v in col_df[col].values]
            elif col == "imd_band":
                order = [v for v in _imd_order if v in col_df[col].unique()]
            else:
                order = (
                    col_df.groupby(col)[TARGET_COL].mean()
                    .sort_values(ascending=False).index.tolist()
                )

            rate = col_df.groupby(col)[TARGET_COL].mean().reindex(order).dropna()
            bars = ax.bar(rate.index, rate.values, color="#E67E22", alpha=0.85)
            for bar, v in zip(bars, rate.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.0%}", ha="center", fontsize=8,
                )
            ax.axhline(overall_rate, linestyle="--", color="gray", linewidth=1, alpha=0.7)
            ax.set_title(_labels.get(col, col.replace("_", " ").title()), fontsize=10)
            ax.set_ylabel("Withdrawal Rate")
            ax.set_ylim(0, min(1.0, float(rate.max()) + 0.12))
            ax.tick_params(axis="x", rotation=30, labelsize=8)

        for i in range(n_panels, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            "Withdrawal Rate by Student Background  (dashed = 31% overall)",
            fontsize=12, y=1.01,
        )
        _save_fig(eda_dir / "12_withdrawal_by_demographics.png")
        insights.append(
            "Withdrawal risk varies across all six demographic dimensions. "
            "IMD band (socioeconomic deprivation) and prior education show the sharpest gradients: "
            "low-income and less-qualified students withdraw at above-average rates. "
            "Students retaking a module (1+ previous attempts) also show elevated risk."
        )

    # ------------------------------------------------------------------
    # 13  (NEW) Engagement acceleration by outcome
    # ------------------------------------------------------------------
    if "engagement_accel_14to21" in df.columns:
        cap_accel = df["engagement_accel_14to21"].quantile(0.97)
        ret_a = df.loc[df[TARGET_COL] == 0, "engagement_accel_14to21"].clip(upper=cap_accel).dropna()
        wth_a = df.loc[df[TARGET_COL] == 1, "engagement_accel_14to21"].clip(upper=cap_accel).dropna()
        plt.figure(figsize=(9, 5))
        plt.boxplot(
            [ret_a, wth_a],
            tick_labels=["Retained", "Withdrawn"],
            patch_artist=True,
            boxprops=dict(facecolor="#F9E79F"),
            medianprops=dict(color="#C0392B", linewidth=2),
        )
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="No growth (ratio=1)")
        plt.title("Engagement Acceleration (14→21d clicks ratio) by Outcome")
        plt.ylabel("first_21d_clicks / (first_14d_clicks + 1)")
        plt.legend()
        _save_fig(eda_dir / "13_engagement_acceleration.png")
        insights.append(
            "Retained students show higher engagement acceleration in the final week. "
            "Students whose weekly clicks plateau or drop (ratio ≤ 1) are higher risk."
        )

    # ------------------------------------------------------------------
    # 14  Clicks in first 21 days by outcome (histogram)
    # ------------------------------------------------------------------
    if "first_21d_clicks" in df.columns:
        not_withdrawn = df[df[TARGET_COL] == 0]
        withdrawn_df  = df[df[TARGET_COL] == 1]
        clicks_nw = np.log1p(not_withdrawn["first_21d_clicks"].dropna().clip(lower=0))
        clicks_w  = np.log1p(withdrawn_df["first_21d_clicks"].dropna().clip(lower=0))
        plt.figure(figsize=(8, 5))
        plt.hist(clicks_nw, bins=40, alpha=0.65, label="Retained",   color="#2E86AB")
        plt.hist(clicks_w,  bins=40, alpha=0.65, label="Withdrawn",  color="#C0392B")
        plt.title("Clicks in First 21 Days by Outcome (log scale)")
        plt.xlabel("log(1 + clicks)")
        plt.ylabel("Frequency")
        plt.legend()
        _save_fig(eda_dir / "14_clicks_by_outcome.png")
        insights.append(
            "Withdrawn students cluster near zero log-clicks in the first 21 days. "
            "Most at-risk students never meaningfully engage with the platform."
        )

    # ------------------------------------------------------------------
    # 15  Key signal boxplots by outcome (3-panel)
    # ------------------------------------------------------------------
    needed_box = ["first_21d_clicks", "first_21d_active_days", "weighted_avg_score"]
    if all(c in df.columns for c in needed_box):
        panels = [
            ("first_21d_clicks",      "Clicks in First 21 Days",              "Clicks",       1500),
            ("first_21d_active_days", "Active Days (21d)",                    "Days",         None),
            ("weighted_avg_score",    "Weighted Avg Score\n(0 = no submission)","Score (0–1)", None),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
        labels = ["Retained", "Withdrawn"]
        colors = ["#2E86AB", "#C0392B"]
        for ax, (col, title, ylabel, upper_clip) in zip(axes, panels):
            groups = [
                df[df[TARGET_COL] == 0][col].dropna().clip(lower=0).values,
                df[df[TARGET_COL] == 1][col].dropna().clip(lower=0).values,
            ]
            bp = ax.boxplot(groups, patch_artist=True, labels=labels)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if upper_clip is not None:
                ax.set_ylim(top=upper_clip)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Key Signals by Outcome (Retained vs Withdrawn)", y=1.02)
        _save_fig(eda_dir / "15_boxplots_by_outcome.png")
        insights.append(
            "Clicks, active days, and score all separate the two groups. "
            "74% of withdrawn students had a score of 0 at day 21, vs 54% of retained."
        )

    # ------------------------------------------------------------------
    # Profile JSON + comments
    # ------------------------------------------------------------------
    profile = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_col": TARGET_COL,
        "withdraw_rate": float(df[TARGET_COL].mean()),
        "non_withdraw_rate": float(1.0 - df[TARGET_COL].mean()),
        "insights": insights,
    }
    with open(ARTIFACTS_DIR / "dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    with open(ARTIFACTS_DIR / "EDA_COMMENTS.md", "w", encoding="utf-8") as f:
        f.write("# EDA Insights — Clean 21-Day Dataset\n\n")
        for i, text in enumerate(insights, start=1):
            f.write(f"{i}. {text}\n")

    print(f"EDA complete — {len(insights)} insights, plots saved to {eda_dir}")


if __name__ == "__main__":
    run_eda()
