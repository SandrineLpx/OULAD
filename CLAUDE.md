# Project Instructions for Claude Code

Before making any changes, read `agents.md` in this directory. It contains:
- The full assignment rubric (MSIS 522 HW1, 100 pts)
- Hard rules that must never be violated
- Current status of each rubric item
- Key design decisions and their rationale
- Project structure and data lineage

## Critical rules (summary — full details in agend.md)

- Do NOT retrain models in the Streamlit app. Load pre-saved files only.
- Do NOT hide the correlation heatmap in an expander (Tab 2, Part 1.4, 5 pts).
- Do NOT hide SHAP global plots behind a prediction click (Tab 4, always visible).
- Do NOT delete `data/processed/studentVle_aggregated_21d.csv` or `data/processed/studentAssessment_aggregated_21d.csv` — they are irreplaceable pre-computed inputs.
- Do NOT add target-leaky features (anything using post-day-21 data, `date_unregistration`, whole-course click aggregates).
- No emojis in UI text, section titles, or captions.
- random_state=42 everywhere.
