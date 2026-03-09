# OULAD Project — Agent Context & Rules

## Assignment

**MSIS 522 HW1 — Foster School of Business, UW**
Individual assignment, 100 points total + 1 bonus point.
Instructor: Prof. Léonard Boussioux
Deliverables: GitHub repo + deployed Streamlit app (public URL required).

---

## Dataset

**Open University Learning Analytics Dataset (OULAD)**
- 32,593 student-module enrolments, 7 modules, 4 presentations (2013B/J, 2014B/J)
- Binary classification target: `withdrawn_flag` = 1 if `final_result == "Withdrawn"` (31.2% positive rate)
- Features restricted to the **first 21 days** of each module presentation (early warning system)

### Data lineage
```
data/raw/studentInfo.csv          ─┐
data/raw/studentRegistration.csv   │
data/raw/courses.csv               ├─► src/build_clean_dataset.py ─► data/oulad_modeling_table_clean.csv
data/processed/studentVle_aggregated_21d.csv      ─┤
data/processed/studentAssessment_aggregated_21d.csv─┘
```
The two `_21d.csv` files are **pre-computed intermediates** — no script in the repo generates them. Do NOT delete them.

---

## Rubric — strict compliance requirements

### Part 1: Descriptive Analytics (25 pts)
| Item | Pts | Where in app | Status |
|---|---|---|---|
| 1.1 Dataset intro (what/where/why, basic stats) | 5 | Tab 1 + Tab 5 About | ✓ |
| 1.2 Target distribution (plot + comment on balance) | 5 | Tab 2 | ✓ |
| 1.3 Feature distributions (≥4 visualizations + interpretation each) | 10 | Tab 2 | ✓ |
| 1.4 Correlation heatmap (plot + comment on strongest correlations) | 5 | Tab 2 — always visible | ✓ |

### Part 2: Predictive Analytics (45 pts)
| Item | Pts | Where in app | Status |
|---|---|---|---|
| 2.2 Logistic Regression baseline | 5 | Tab 3 model comparison table | ✓ |
| 2.3 Decision Tree with 5-fold CV | 5 | Tab 3 | ✓ |
| 2.4 Random Forest with 5-fold CV | 10 | Tab 3 | ✓ |
| 2.5 Boosted Trees (LightGBM) with 5-fold CV | 10 | Tab 3 | ✓ |
| 2.6 Neural Network MLP (≥2 hidden layers, Keras) | 10 | Tab 3 — NN expander | ✓ |
| 2.7 Model comparison: table + bar chart + paragraph (best model, surprises, tradeoffs) | 5 | Tab 3 | ✓ |

### Part 3: Explainability (10 pts)
| Item | Required | Where in app | Status |
|---|---|---|---|
| SHAP beeswarm summary plot | ✓ | Tab 4 — "SHAP summary" expander (always visible) | ✓ |
| SHAP bar plot (mean |SHAP|) | ✓ | Tab 4 — always visible below form | ✓ |
| SHAP waterfall for one prediction | ✓ | Tab 4 — local waterfall (appears after clicking Predict) | ✓ |
| Written interpretation (strongest features, direction, decision-maker utility) | ✓ | Tab 4 — interpretation guide expander | ✓ |

### Part 4: Streamlit Deployment (20 pts)
| Tab | Pts | Must contain |
|---|---|---|
| Tab 1 Executive Summary | 4 | Dataset description, "so what", approach overview, key findings. Non-technical audience. |
| Tab 2 Descriptive Analytics | 4 | All Part 1 visualizations with captions. Target dist, feature dists, correlation heatmap. |
| Tab 3 Model Performance | 4 | Comparison table + bar chart (2.7), ROC curves, best hyperparameters per model, all Part 2 metrics. |
| Tab 4 Explainability & Interactive Prediction | 8 | SHAP summary + bar plots, interactive sliders/dropdowns, model selector, predicted class + probability, SHAP waterfall for custom input. |

### Bonus (1 pt)
- Neural network hyperparameter tuning (grid search over hidden layer sizes, learning rates, dropout) with visualization of results.

---

## Chart sizing standard — benchmark chart

**Benchmark:** the SHAP global bar chart (`shap_bar.png`) displayed in Tab 4.

| Property | Value | Notes |
|---|---|---|
| Figure size | `(7, 5.5)` inches | Set in `src/explainability.py` |
| DPI | `130` | All artifact PNGs |
| Streamlit display | `st.columns([0.65, 0.35])` — chart in left column | Prevents full-width stretch |
| Font sizes | `rcParams`: title 11pt, labels 10pt, ticks 9pt, legend 9pt | Set at top of each generator function |

**Rules for all charts:**
- Single-panel artifacts: `figsize=(7–8, 5–5.5)`, DPI=130
- Multi-panel artifacts (2–3 side-by-side): `figsize=(11, 4–4.5)`, DPI=130, full-width OK (panels are naturally smaller)
- Live Streamlit `st.pyplot()` charts: `figsize=(6.5, 4.5)` max, wrapped in `st.columns([0.65, 0.35])`
- **Never** use `use_container_width=True` on a full-page-width single chart — always constrain with a column
- `rcParams.update(...)` must be set at the top of every artifact-generating function (`run_eda`, `generate_shap_artifacts`, etc.)

---

## Hard rules — never violate

1. **No retraining in the app** — Streamlit loads pre-saved models only (`models/*.joblib`, `models/nn_model.keras`). The pipeline (`python -m src.run_pipeline`) does all training offline.
2. **random_state=42** everywhere randomness is involved.
3. **No target leakage** — all features restricted to days 0–21. `date_unregistration` is the withdrawal event itself — always dropped. Whole-course aggregates (`total_clicks`, 18 `clicks_activity_*` columns) are excluded.
4. **No student-level contamination** — `GroupShuffleSplit` + `StratifiedGroupKFold` by `id_student` (3,538 students appear in >1 enrolment).
5. **SHAP plots must be visible without interaction** — the grader should see the global bar chart and beeswarm on Tab 4 load, without needing to click Predict.
6. **Correlation heatmap must be always visible** on Tab 2 — not hidden in an expander (Part 1.4, 5 pts).
7. **Deployed public URL required** — localhost links are not accepted. Deploy to Streamlit Cloud or equivalent.
8. **No emojis** in section titles, labels, or captions.

---

## Project structure

```
.
├── data/
│   ├── raw/                    # original OULAD CSV files (do not modify)
│   ├── processed/              # pre-computed 21d aggregates (do not delete)
│   └── oulad_modeling_table_clean.csv
├── src/
│   ├── config.py               # paths, DROP_COLS, constants
│   ├── build_clean_dataset.py  # joins raw + processed into modeling table
│   ├── feature_engineering.py  # trajectories, flags, z-scores
│   ├── data_preprocessing.py   # GroupShuffleSplit, StratifiedGroupKFold
│   ├── eda.py                  # 15 EDA charts → artifacts/eda/
│   ├── train_models.py         # 8 sklearn models + threshold + calibration
│   ├── neural_network.py       # Keras 3 MLP (PyTorch backend)
│   ├── evaluate_models.py      # holdout metrics, subgroup, calibration curve
│   ├── explainability.py       # SHAP global/local/group
│   └── run_pipeline.py         # orchestrator (run this to regenerate all artifacts)
├── app_utils/
│   ├── constants.py            # MODEL_DISPLAY_NAMES, FRIENDLY_LABELS, EDA_NOTES, MODEL_NOTES
│   ├── ui.py                   # CSS, render_header, render_metric_card, open/close_section_card, show_image_card
│   ├── data.py                 # artifact loaders (load_csv, load_available_models, load_schema, etc.)
│   ├── forms.py                # render_input_form, build_default_row
│   └── charts.py               # reusable chart helpers
├── streamlit_app.py            # 5-tab app (Executive Summary / Descriptive Analytics / Model Performance / Prediction & Explainability / About)
├── models/                     # saved model files (generated by pipeline)
├── artifacts/                  # charts, CSVs, JSON reports (generated by pipeline)
│   └── eda/                    # EDA plots (01–15_*.png)
├── docs/
│   └── MSIS_522_HW1_Data_Science_Workflow.docx.md  # original rubric
├── README.md
├── agents.md                   # this file
└── requirements.txt
```

---

## Key design decisions (do not reverse without strong reason)

| Decision | Rationale |
|---|---|
| 21-day window for all features | First TMA due at day 19 (module AAA); withdrawal decisions appear later (week 4+) |
| PR-AUC as primary selection metric | 31.2% class imbalance makes ROC-AUC misleading; PR-AUC penalises false positives on minority class |
| GroupShuffleSplit instead of random split | 3,538 students in >1 enrolment — random split would leak student patterns |
| Temporal holdout: train 2013, test 2014 | Validates generalisation to a new cohort; withdrawal rate shift from ~28% to ~34% is real |
| Isotonic calibration on best model | Ensures predicted probabilities are trustworthy (Brier score reported) |
| `is_feb_start` extracted from `code_presentation` | Captures semester effect without encoding the year (which would cause temporal leakage) |
| `code_presentation` kept for joins but excluded from model features | Avoids year-encoding leakage while preserving temporal holdout logic |

---

## Streamlit app — tab layout summary

| Tab | Key sections | Notes |
|---|---|---|
| 1 Executive Summary | The problem, the data (+ module chart), the approach (+ safeguards), results (3 KPI cards), key findings (3 cards), recommended actions (3 cards) | Non-technical narrative; no jargon |
| 2 Descriptive Analytics | Dataset summary line, inner tabs: Early signal overview (target dist + clicks + heatmap always visible), Key signals by outcome (boxplots + demographics + Additional EDA expander with 4 plots) | Heatmap must stay out of expander |
| 3 Model Performance | Model snapshot (3 KPIs), inner tabs: Overview (bar chart + holdout table + ROC curves + 2.7 paragraph), Diagnostics (PR/CM/threshold + calibration/temporal + subgroup expander), Technical details (CV table + hyperparams + DT viz + NN training curve) | 2.7 paragraph covers best model / surprises / tradeoffs |
| 4 Prediction & Explainability | Global SHAP bar + beeswarm (always visible), interpretation guide (expanded), SHAP case study expander, SHAP group contribution expander, interactive form (model selector + student profile + Predict), result card + gauge, live SHAP waterfall | Global SHAP always visible even before prediction |
| 5 About | Dataset card, source tables, target variable, 6-step pipeline (expanders), reproducibility commands | Reference only; no charts |

---

## Run commands

```bash
# Regenerate all artifacts and models
python -m src.run_pipeline

# Regenerate EDA charts only
python -m src.eda

# Regenerate models only
python -m src.train_models

# Launch app
streamlit run streamlit_app.py
```
