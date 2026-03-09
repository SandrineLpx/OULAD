"""
neural_network.py
=================
Keras MLP for the OULAD withdrawal prediction task.

Uses Keras 3 with PyTorch backend (set via KERAS_BACKEND env var).
Falls back gracefully if neither Keras nor a backend is available.

Architecture
------------
  Input (scaled numeric + OHE categorical)
    Dense(128, relu) + BatchNormalization + Dropout(0.3)
    Dense(64,  relu) + BatchNormalization + Dropout(0.3)
    Dense(32,  relu)
    Dense(1,   sigmoid)

Training strategy
-----------------
* Preprocessing: same sklearn ColumnTransformer as tree models (median impute + scale numeric).
* Class imbalance: handled via class_weight passed to model.fit().
* 5-fold StratifiedGroupKFold CV (same fold scheme as sklearn models).
* EarlyStopping on val_pr_auc (patience=15) + ReduceLROnPlateau.
* Final model trained on full training split, evaluated on holdout.
* Appends a row to model_comparison.csv for apples-to-apples comparison.
"""
from __future__ import annotations

import json
import os
import warnings

# Set Keras backend BEFORE importing keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from src.config import ARTIFACTS_DIR, MODELS_DIR, RANDOM_STATE
from src.data_preprocessing import (
    build_modeling_dataset,
    group_train_test_split,
    load_modeling_table,
    make_preprocessor,
)

warnings.filterwarnings("ignore")


def _check_keras():
    """Check that Keras 3 and a backend (torch) are available."""
    try:
        import keras
        # Verify backend is functional
        _ = keras.backend.backend()
        return keras
    except (ImportError, Exception):
        return None


def _build_model(keras_mod, input_dim: int, dropout_rate: float = 0.3):
    """Build and compile a Keras MLP."""
    keras_mod.utils.set_random_seed(RANDOM_STATE)

    model = keras_mod.Sequential([
        keras_mod.layers.Input(shape=(input_dim,)),
        keras_mod.layers.Dense(128, activation="relu"),
        keras_mod.layers.BatchNormalization(),
        keras_mod.layers.Dropout(dropout_rate),
        keras_mod.layers.Dense(64, activation="relu"),
        keras_mod.layers.BatchNormalization(),
        keras_mod.layers.Dropout(dropout_rate),
        keras_mod.layers.Dense(32, activation="relu"),
        keras_mod.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras_mod.optimizers.Adam(learning_rate=1e-3),
        loss=keras_mod.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=[
            keras_mod.metrics.AUC(name="pr_auc", curve="PR"),
            keras_mod.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model


def _get_callbacks(keras_mod):
    return [
        keras_mod.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=15, restore_best_weights=True, verbose=0,
        ),
        keras_mod.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=6, min_lr=1e-6, verbose=0,
        ),
    ]


def _evaluate_nn(y_true, y_proba, label: str = "") -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    tag = f"_{label}" if label else ""
    return {
        f"accuracy{tag}": float(accuracy_score(y_true, y_pred)),
        f"precision{tag}": float(precision_score(y_true, y_pred, zero_division=0)),
        f"recall{tag}": float(recall_score(y_true, y_pred, zero_division=0)),
        f"f1{tag}": float(f1_score(y_true, y_pred, zero_division=0)),
        f"roc_auc{tag}": float(roc_auc_score(y_true, y_proba)),
        f"pr_auc{tag}": float(average_precision_score(y_true, y_proba)),
        f"brier{tag}": float(brier_score_loss(y_true, y_proba)),
    }


def train_neural_network() -> None:
    keras_mod = _check_keras()
    if keras_mod is None:
        print("Keras 3 not installed or no backend available — skipping neural network. "
              "Install with: pip install keras torch")
        return

    backend_name = keras_mod.backend.backend()
    print(f"Using Keras {keras_mod.__version__} with {backend_name} backend")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    raw_df = load_modeling_table()
    bundle = build_modeling_dataset(raw_df)
    X_train, X_test, y_train, y_test, groups_train, _ = group_train_test_split(bundle)

    # Preprocess (scale + OHE)
    preprocessor = make_preprocessor(bundle.numeric_cols, bundle.categorical_cols, scale_numeric=True)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
        X_test_trans  = X_test_trans.toarray()
    X_train_trans = X_train_trans.astype(np.float32)
    X_test_trans  = X_test_trans.astype(np.float32)
    y_train_arr = y_train.values.astype(np.float32)
    y_test_arr  = y_test.values.astype(np.float32)

    # Class weights
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_train_arr)
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}

    input_dim = X_train_trans.shape[1]

    # ------------------------------------------------------------------
    # 5-fold cross-validation
    # ------------------------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=5)
    fold_metrics: list[dict] = []
    histories: list = []

    print(f"Neural network CV (5-fold)  input_dim={input_dim}")
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train)):
        X_tr, X_val = X_train_trans[tr_idx], X_train_trans[val_idx]
        y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]

        model = _build_model(keras_mod, input_dim)
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=512,
            class_weight=class_weight_dict,
            callbacks=_get_callbacks(keras_mod),
            verbose=0,
        )
        histories.append(history.history)

        y_proba_val = model.predict(X_val, verbose=0).ravel()
        m = _evaluate_nn(y_val, y_proba_val)
        m["fold"] = fold_i + 1
        fold_metrics.append(m)
        print(f"  Fold {fold_i+1}: PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}")

    cv_pr_auc  = float(np.mean([m["pr_auc"] for m in fold_metrics]))
    cv_pr_std  = float(np.std([m["pr_auc"] for m in fold_metrics]))
    cv_f1_mean = float(np.mean([m["f1"] for m in fold_metrics]))
    cv_f1_std  = float(np.std([m["f1"] for m in fold_metrics]))
    print(f"  CV PR-AUC = {cv_pr_auc:.4f} ± {cv_pr_std:.4f}")

    # ------------------------------------------------------------------
    # Final model on full training set
    # ------------------------------------------------------------------
    final_model = _build_model(keras_mod, input_dim)
    # Use 10% of training data as validation for early stopping
    val_size = int(0.1 * len(X_train_trans))
    X_val_final, y_val_final = X_train_trans[-val_size:], y_train_arr[-val_size:]
    X_fit, y_fit = X_train_trans[:-val_size], y_train_arr[:-val_size]

    final_history = final_model.fit(
        X_fit, y_fit,
        validation_data=(X_val_final, y_val_final),
        epochs=200,
        batch_size=512,
        class_weight=class_weight_dict,
        callbacks=_get_callbacks(keras_mod),
        verbose=1,
    )

    y_proba_test = final_model.predict(X_test_trans, verbose=0).ravel()
    holdout_metrics = _evaluate_nn(y_test_arr, y_proba_test)

    # ------------------------------------------------------------------
    # Save model and artifacts
    # ------------------------------------------------------------------
    model_save_path = MODELS_DIR / "nn_model.keras"
    final_model.save(str(model_save_path))
    print(f"Neural network saved: {model_save_path}")

    nn_results = {
        "model": "neural_network_keras",
        "architecture": "Dense(128,relu)+BN+Dropout -> Dense(64)+BN+Dropout -> Dense(32) -> sigmoid",
        "loss": "BinaryFocalCrossentropy(gamma=2.0)",
        "cv_pr_auc_mean": cv_pr_auc,
        "cv_pr_auc_std": cv_pr_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "class_weight": class_weight_dict,
        "input_dim": input_dim,
        "fold_metrics": fold_metrics,
        "holdout": holdout_metrics,
    }
    with open(ARTIFACTS_DIR / "nn_cv_results.json", "w") as f:
        json.dump(nn_results, f, indent=2)

    holdout_dict = {"model": "neural_network_keras", **holdout_metrics}
    with open(ARTIFACTS_DIR / "nn_holdout_metrics.json", "w") as f:
        json.dump(holdout_dict, f, indent=2)

    # ------------------------------------------------------------------
    # Append NN row to model_comparison.csv for apples-to-apples view
    # ------------------------------------------------------------------
    comp_path = ARTIFACTS_DIR / "model_comparison.csv"
    nn_row = {
        "model": "neural_network_keras",
        "cv_pr_auc_mean": cv_pr_auc,
        "cv_pr_auc_std": cv_pr_std,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "cv_accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "cv_recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
        "cv_roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
    }
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[comp_df["model"] != "neural_network_keras"]
        nn_df   = pd.DataFrame([nn_row])
        comp_df = pd.concat([comp_df, nn_df], ignore_index=True)
        comp_df.to_csv(comp_path, index=False)

    # ------------------------------------------------------------------
    # Training curve plot
    # ------------------------------------------------------------------
    _plot_training_curve(final_history.history)

    print(f"\nNeural network complete.")
    print(f"  Holdout PR-AUC : {holdout_metrics['pr_auc']:.4f}")
    print(f"  Holdout F1     : {holdout_metrics['f1']:.4f}")
    print(f"  Holdout Recall : {holdout_metrics['recall']:.4f}")


def _plot_training_curve(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["loss"], label="Train", color="#2E86AB")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val", color="#C0392B")
    axes[0].set_title("Training Loss (Binary Focal Crossentropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # PR-AUC — Keras 3 metric name is "pr_auc"
    auc_key = next((k for k in ("pr_auc", "auc") if k in history), None)
    val_auc_key = next((k for k in ("val_pr_auc", "val_auc") if k in history), None)
    if auc_key:
        axes[1].plot(history[auc_key], label="Train PR-AUC", color="#2E86AB")
    if val_auc_key:
        axes[1].plot(history[val_auc_key], label="Val PR-AUC", color="#C0392B")
    axes[1].set_title("PR-AUC During Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].legend()

    plt.suptitle("Neural Network Training Curve", fontsize=13)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "nn_training_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train_neural_network()
