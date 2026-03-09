"""
data_preprocessing.py
======================
Data loading, preprocessing pipeline, and train/test splitting.

Key changes from previous version
----------------------------------
* `load_modeling_table` now defaults to DATA_CLEAN_PATH (leak-free dataset).
* Train/test split uses GroupShuffleSplit keyed on id_student so the same
  student never appears in both train and test across enrollments.
* Cross-validation uses StratifiedGroupKFold for the same reason.
* `temporal_split` added for out-of-time validation (train 2013, test 2014).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_CLEAN_PATH, DROP_COLS, RANDOM_STATE, STUDENT_ID_COL, TARGET_COL
from src.feature_engineering import add_derived_features


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    student_ids: pd.Series          # for GroupShuffleSplit / StratifiedGroupKFold
    target_col: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    raw_df: pd.DataFrame = field(repr=False)   # full df incl. code_presentation for temporal split


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_modeling_table(path: str | Path | None = None) -> pd.DataFrame:
    """Load the clean 21-day modeling table. Defaults to DATA_CLEAN_PATH."""
    data_path = DATA_CLEAN_PATH if path is None else Path(path)
    return pd.read_csv(data_path)


def build_modeling_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> DatasetBundle:
    """
    Apply feature engineering, drop leakage/ID/redundant columns, separate X/y.

    Parameters
    ----------
    df : DataFrame  The raw modeling table (from load_modeling_table).
    target_col : str  Name of the binary target column.

    Returns
    -------
    DatasetBundle
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    out = add_derived_features(df)

    # Extract student IDs before dropping them (needed for group splits)
    student_ids = out[STUDENT_ID_COL].copy() if STUDENT_ID_COL in out.columns else pd.Series(
        np.arange(len(out)), name=STUDENT_ID_COL
    )

    drop_cols = [c for c in DROP_COLS if c in out.columns]
    feature_df = out.drop(columns=drop_cols + [target_col], errors="ignore")
    y = out[target_col].astype(int)

    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    return DatasetBundle(
        X=feature_df,
        y=y,
        student_ids=student_ids,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        raw_df=out,
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def group_train_test_split(
    bundle: DatasetBundle,
    test_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Student-aware 80/20 split.  The same student never appears in both train
    and test sets (prevents student-level memorisation).

    Returns
    -------
    X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(
        gss.split(bundle.X, bundle.y, groups=bundle.student_ids)
    )
    return (
        bundle.X.iloc[train_idx].reset_index(drop=True),
        bundle.X.iloc[test_idx].reset_index(drop=True),
        bundle.y.iloc[train_idx].reset_index(drop=True),
        bundle.y.iloc[test_idx].reset_index(drop=True),
        bundle.student_ids.iloc[train_idx].reset_index(drop=True),
        bundle.student_ids.iloc[test_idx].reset_index(drop=True),
    )


def temporal_split(
    bundle: DatasetBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Out-of-time validation: train on 2013 presentations, test on 2014.
    Withdrawal rates shifted from ~28% (2013) to ~34% (2014), making this
    a realistic test of temporal generalization.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    pres = bundle.raw_df["code_presentation"] if "code_presentation" in bundle.raw_df.columns else None
    if pres is None:
        raise ValueError("code_presentation not found in raw_df for temporal split.")
    train_mask = pres.str.startswith("2013").values
    return (
        bundle.X[train_mask].reset_index(drop=True),
        bundle.X[~train_mask].reset_index(drop=True),
        bundle.y[train_mask].reset_index(drop=True),
        bundle.y[~train_mask].reset_index(drop=True),
    )


def get_cv(n_splits: int = 5) -> StratifiedGroupKFold:
    """Return a StratifiedGroupKFold CV object (pass groups= to cross_validate)."""
    return StratifiedGroupKFold(n_splits=n_splits)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def make_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Sklearn ColumnTransformer:
      numeric  -> median imputation -> (optional) StandardScaler
      categorical -> most-frequent imputation -> OneHotEncoder
    Tree models should pass scale_numeric=False.
    """
    num_steps: list[Any] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Feature schema (for Streamlit prediction UI)
# ---------------------------------------------------------------------------

def save_feature_schema(X: pd.DataFrame, output_path: str | Path) -> None:
    schema: dict[str, dict] = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            schema[col] = {
                "type": "numeric",
                "default": float(X[col].median()) if X[col].notna().any() else 0.0,
                "min": float(X[col].quantile(0.01)) if X[col].notna().any() else 0.0,
                "max": float(X[col].quantile(0.99)) if X[col].notna().any() else 1.0,
            }
        else:
            values = (
                X[col].fillna("missing").astype(str)
                .value_counts().head(15).index.tolist()
            )
            schema[col] = {
                "type": "categorical",
                "default": values[0] if values else "missing",
                "choices": values if values else ["missing"],
            }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
