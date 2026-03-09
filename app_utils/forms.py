from __future__ import annotations

import math
from typing import Any

import streamlit as st

from app_utils.ui import pretty_name


def _muted_line(text: str) -> None:
    st.markdown(f'<div class="small-muted">{text}</div>', unsafe_allow_html=True)


def _is_decimal_feature(feature: str) -> bool:
    return any(
        key in feature
        for key in [
            "rate",
            "ratio",
            "share",
            "avg",
            "score",
            "delay",
            "intensity",
            "density",
        ]
    )


def render_numeric_input(feature: str, meta: dict[str, Any]) -> float | int:
    min_v = float(meta.get("min", 0.0))
    max_v = float(meta.get("max", max(min_v + 1.0, 1.0)))
    default = float(meta.get("default", min_v))

    if _is_decimal_feature(feature):
        is_unit_interval = min_v >= 0.0 and max_v <= 1.0
        step = 0.01 if is_unit_interval or any(k in feature for k in ["rate", "ratio", "share"]) else 0.1
        fmt = "%.2f" if step == 0.01 else "%.1f"
        value = float(min(max(default, min_v), max_v))
        return st.number_input(
            label=pretty_name(feature),
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(round(value, 2 if step == 0.01 else 1)),
            step=float(step),
            format=fmt,
        )

    min_i = int(math.floor(min_v))
    max_i = int(math.ceil(max_v))
    default_i = int(round(default))
    default_i = min(max(default_i, min_i), max_i)
    return st.number_input(
        label=pretty_name(feature),
        min_value=min_i,
        max_value=max_i,
        value=default_i,
        step=1,
        format="%d",
    )


def render_main_numeric_input(feature: str, meta: dict[str, Any]) -> float | int:
    slider_features = {
        "studied_credits",
        "first_21d_clicks",
        "first_21d_active_days",
        "submission_rate",
        "weighted_avg_score",
    }
    if feature not in slider_features:
        return render_numeric_input(feature, meta)

    min_v = float(meta.get("min", 0.0))
    max_v = float(meta.get("max", max(min_v + 1.0, 1.0)))
    default = float(meta.get("default", min_v))

    if feature in {"studied_credits", "first_21d_clicks", "first_21d_active_days"}:
        min_i = int(math.floor(min_v))
        max_i = int(math.ceil(max_v))
        default_i = int(round(default))
        default_i = min(max(default_i, min_i), max_i)
        if feature == "first_21d_clicks":
            step = max(1, int((max_i - min_i) / 200))
        elif feature == "studied_credits":
            step = 5
        else:
            step = 1
        return st.slider(
            label=pretty_name(feature),
            min_value=min_i,
            max_value=max_i,
            value=default_i,
            step=step,
        )

    if feature == "submission_rate":
        value = float(min(max(default, min_v), max_v))
        return st.slider(
            label=pretty_name(feature),
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(round(value, 2)),
            step=0.01,
            format="%.2f",
        )

    if feature == "weighted_avg_score":
        value = float(min(max(default, min_v), max_v))
        return st.slider(
            label=pretty_name(feature),
            min_value=float(min_v),
            max_value=float(max_v),
            value=float(round(value, 1)),
            step=0.1,
            format="%.1f",
        )

    value = float(min(max(default, min_v), max_v))
    return st.slider(
        label=pretty_name(feature),
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(round(value, 1)),
        step=0.5,
        format="%.1f",
    )


def build_default_row(schema: dict[str, Any]) -> dict[str, Any]:
    row = {}
    for feature, meta in schema.items():
        row[feature] = meta.get("default", 0)
    return row


def render_input_form(schema: dict[str, Any], n_cols: int = 2) -> dict[str, Any]:
    main_candidates = [
        "code_module",
        "highest_education",
        "age_band",
        "studied_credits",
        "first_21d_clicks",
        "first_21d_active_days",
        "submission_rate",
        "weighted_avg_score",
    ]
    key_candidates = [
        "code_module",
        "highest_education",
        "imd_band",
        "age_band",
        "studied_credits",
        "num_of_prev_attempts",
        "first_21d_clicks",
        "first_21d_active_days",
        "submission_rate",
        "weighted_avg_score",
        "late_submission_rate",
        "engagement_intensity_21d",
    ]

    selected_features = [f for f in key_candidates if f in schema]
    main_features = [f for f in main_candidates if f in selected_features]
    advanced_features = [f for f in selected_features if f not in main_features]
    values: dict[str, Any] = {}

    st.markdown("**Student profile inputs**")
    n_main_cols = max(1, int(n_cols))
    main_cols = st.columns(n_main_cols)
    for i, feature in enumerate(main_features):
        meta = schema[feature]
        target_col = main_cols[i % n_main_cols]
        with target_col:
            if meta.get("type") == "numeric":
                values[feature] = render_main_numeric_input(feature, meta)
            else:
                choices = [str(c) for c in meta.get("choices", ["missing"])]
                default = str(meta.get("default", choices[0]))
                idx = choices.index(default) if default in choices else 0
                values[feature] = st.selectbox(pretty_name(feature), choices, index=idx)

    if advanced_features:
        with st.expander("Advanced inputs", expanded=False):
            helper_col, reset_col = st.columns([0.78, 0.22])
            with helper_col:
                _muted_line(
                    "Optional details for sensitivity checks. Adjust values to see how the predicted risk changes."
                )
            with reset_col:
                reset_advanced = st.form_submit_button(
                    "Reset advanced inputs",
                    use_container_width=True,
                )

            advanced_set = set(advanced_features)
            handled_features: set[str] = set()

            # Dashboard-friendly labels for the advanced scenario-testing section.
            adv_labels = {
                "imd_band": "Socioeconomic background (IMD band)",
                "num_of_prev_attempts": "Prior attempts at this module",
                "late_submission_rate": "Late submission rate (first 21 days)",
                "engagement_intensity_21d": "Average daily engagement (first 21 days)",
            }

            # ---------- Defaults and state keys ----------
            imd_choices = [str(c) for c in schema.get("imd_band", {}).get("choices", ["missing"])]
            imd_default = str(
                schema.get("imd_band", {}).get("default", imd_choices[0] if imd_choices else "missing")
            )
            if imd_default not in imd_choices and imd_choices:
                imd_default = imd_choices[0]

            prev_meta = schema.get("num_of_prev_attempts", {})
            prev_min = int(math.floor(float(prev_meta.get("min", 0) or 0)))
            prev_max = int(math.ceil(float(prev_meta.get("max", 10) or 10)))
            if prev_max < prev_min:
                prev_max = prev_min
            prev_default = int(round(float(prev_meta.get("default", prev_min) or prev_min)))
            prev_default = min(max(prev_default, prev_min), prev_max)

            late_default_ratio = float(schema.get("late_submission_rate", {}).get("default", 0.0) or 0.0)
            late_default_ratio = min(max(late_default_ratio, 0.0), 1.0)

            eng_meta = schema.get("engagement_intensity_21d", {})
            eng_min, eng_max = 0.0, 200.0
            eng_default = float(eng_meta.get("default", eng_meta.get("median", 10.0)) or 10.0)
            eng_default = min(max(eng_default, eng_min), eng_max)

            k_imd = "adv_imd_band"
            k_prev = "adv_num_of_prev_attempts"
            k_late_ratio = "adv_late_submission_rate_ratio"
            k_eng = "adv_engagement_intensity_21d"

            if k_imd not in st.session_state:
                st.session_state[k_imd] = imd_default
            if k_prev not in st.session_state:
                st.session_state[k_prev] = prev_default
            if k_late_ratio not in st.session_state:
                st.session_state[k_late_ratio] = late_default_ratio
            if k_eng not in st.session_state:
                st.session_state[k_eng] = eng_default

            if reset_advanced:
                st.session_state[k_imd] = imd_default
                st.session_state[k_prev] = prev_default
                st.session_state[k_late_ratio] = late_default_ratio
                st.session_state[k_eng] = eng_default

            def _format_imd_choice(raw: str) -> str:
                return raw.replace("_", " ").strip()

            # ---------- Group 1: Background ----------
            if "imd_band" in advanced_set:
                st.markdown("**Background**")
                _muted_line("Context variable for segmentation and scenario testing.")
                bg_col, _ = st.columns(2)
                with bg_col:
                    idx = (
                        imd_choices.index(st.session_state[k_imd])
                        if st.session_state[k_imd] in imd_choices
                        else 0
                    )
                    values["imd_band"] = st.selectbox(
                        adv_labels["imd_band"],
                        imd_choices,
                        index=idx,
                        format_func=_format_imd_choice,
                        key=k_imd,
                    )
                handled_features.add("imd_band")

            # ---------- Group 2: Academic history ----------
            if {"num_of_prev_attempts", "late_submission_rate"} & advanced_set:
                st.markdown("**Academic history**")
                _muted_line("Use these controls to test prior-attempt and pacing scenarios.")
                a1, a2 = st.columns(2)
                if "num_of_prev_attempts" in advanced_set:
                    with a1:
                        values["num_of_prev_attempts"] = int(
                            st.number_input(
                                adv_labels["num_of_prev_attempts"],
                                min_value=prev_min,
                                max_value=prev_max,
                                value=int(st.session_state[k_prev]),
                                step=1,
                                key=k_prev,
                            )
                        )
                    handled_features.add("num_of_prev_attempts")
                if "late_submission_rate" in advanced_set:
                    with a2:
                        late_ratio = float(
                            st.slider(
                                adv_labels["late_submission_rate"],
                                min_value=0.0,
                                max_value=1.0,
                                value=float(st.session_state[k_late_ratio]),
                                step=0.01,
                                key=k_late_ratio,
                                help="Fraction between 0 and 1.",
                            )
                        )
                        values["late_submission_rate"] = late_ratio
                        _muted_line(f"Current value: {late_ratio:.0%} (stored as {late_ratio:.2f}).")
                    handled_features.add("late_submission_rate")

            # ---------- Group 3: Engagement ----------
            if "engagement_intensity_21d" in advanced_set:
                st.markdown("**Engagement**")
                _muted_line("Derived engagement signal used for sensitivity checks.")
                e_col, _ = st.columns(2)
                with e_col:
                    eng_val = float(
                        st.number_input(
                        adv_labels["engagement_intensity_21d"],
                        min_value=float(eng_min),
                        max_value=float(eng_max),
                        value=float(round(float(st.session_state[k_eng]), 1)),
                        step=0.1,
                        format="%.1f",
                        key=k_eng,
                        help="Clicks per active day (computed).",
                        )
                    )
                    values["engagement_intensity_21d"] = float(eng_val)
                    _muted_line("Units: clicks per active day (computed).")
                handled_features.add("engagement_intensity_21d")

            # Fallback rendering for any additional advanced fields not in the
            # grouped scenario-testing layout.
            remaining_features = [f for f in advanced_features if f not in handled_features]
            if remaining_features:
                st.markdown("**Additional details**")
                r1, r2 = st.columns(2)
                for i, feature in enumerate(remaining_features):
                    meta = schema[feature]
                    target_col = r1 if i % 2 == 0 else r2
                    with target_col:
                        if meta.get("type") == "numeric":
                            values[feature] = render_numeric_input(feature, meta)
                        else:
                            choices = [str(c) for c in meta.get("choices", ["missing"])]
                            default = str(meta.get("default", choices[0]))
                            idx = choices.index(default) if default in choices else 0
                            values[feature] = st.selectbox(pretty_name(feature), choices, index=idx)

    return values
