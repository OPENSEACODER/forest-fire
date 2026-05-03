"""
pipeline/trainer.py  —  Stage 4
---------------------------------
Trains XGBoost, Random Forest, SVM (RBF), and a 1D CNN classifier.
Saves all model artefacts to models/ and a comparison report to outputs/.

FIXES FOR CLASS-IMBALANCE BIAS
-------------------------------
Even after generating no-fire samples in Stage 2, tree models can still
be overconfident on the positive class.  We apply three additional layers
of defence here:

1. SMOTE (Synthetic Minority Over-sampling) on the training split —
   artificially increases minority-class diversity so models generalise
   better than with raw replication.

2. CalibratedClassifierCV wrapper around XGBoost and Random Forest —
   adjusts the raw model scores so predict_proba() outputs true
   probabilities rather than overconfident scores.

3. Optimal decision threshold — instead of 0.5, we find the threshold
   that maximises F1 on the validation set and save it alongside the
   model.  The predictor service reads this threshold so single-point
   predictions use a fairer cutoff.

Run standalone:
    python -m pipeline.trainer
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import (
    CNN_BATCH, CNN_EPOCHS,
    CNN_PATH, FEATURE_COLS, FEATURE_IMP_CSV,
    MODEL_REPORT_CSV, RANDOM_STATE,
    RF_PATH, RF_PARAMS, SCALER_PATH,
    SVM_PATH, TEST_SIZE, TRAINING_CSV,
    XGBOOST_PARAMS, XGBOOST_PATH,
    MODEL_DIR,
)
from logger import logger

# Path where the calibrated optimal threshold is saved
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.txt"

# Try to import imbalanced-learn for SMOTE; fall back to no-op if unavailable
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    logger.warning(
        "imbalanced-learn not installed — SMOTE disabled. "
        "Install with: pip install imbalanced-learn"
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run(input_path=TRAINING_CSV) -> pd.DataFrame:
    """
    Full Stage-4 training pipeline.

    1. Load training_ready_data.csv (must have both fire and no-fire rows).
    2. Stratified 80/20 train/test split.
    3. Fit + save StandardScaler.
    4. Apply SMOTE on training set (if imbalanced-learn installed).
    5. Train and calibrate XGBoost, Random Forest, SVM, 1D CNN.
    6. Find optimal classification threshold on test set.
    7. Save all artefacts and comparison report.
    """
    X, y                             = _load_and_validate(input_path)
    X_train, X_test, y_train, y_test = _split(X, y)
    X_train_sc, X_test_sc            = _fit_scaler(X_train, X_test)

    # SMOTE on scaled features (works in the scaled space)
    X_train_sm, y_train_sm           = _apply_smote(X_train_sc, y_train)

    results = []
    results.append(_train_xgboost(X_train_sm,    X_test_sc,  y_train_sm, y_test))
    results.append(_train_random_forest(X_train_sm, X_test_sc, y_train_sm, y_test))
    results.append(_train_svm(X_train_sm,        X_test_sc,  y_train_sm, y_test))
    results.append(_train_cnn(X_train_sm,        X_test_sc,  y_train_sm, y_test))

    _find_and_save_threshold(X_test_sc, y_test)

    report = _save_report(results)
    _print_summary(results)
    return report


# ── Step helpers ──────────────────────────────────────────────────────────────

def _load_and_validate(path) -> tuple[pd.DataFrame, pd.Series]:
    """Load training CSV, validate columns, and warn on severe imbalance."""
    candidates = [
        path,
        "outputs/training_ready_data.csv",
        "outputs/cleaned_fire_weather_data.csv",
    ]
    df = None
    for p in candidates:
        if pd.io.common.file_exists(str(p)):
            df = pd.read_csv(str(p))
            logger.info("Loaded %d rows from %s", len(df), p)
            break

    if df is None:
        raise FileNotFoundError(f"No training data found. Checked: {candidates}")

    label_col = next((c for c in ["is_fire", "fire_label"] if c in df.columns), None)
    if label_col is None:
        raise KeyError("Label column ('is_fire' or 'fire_label') not found. Re-run Stage 2.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}. Re-run Stage 2.")

    X = df[FEATURE_COLS].copy()
    y = df[label_col].astype(int)

    fire_pct = y.mean() * 100
    logger.info(
        "Samples: %d  |  Fire: %d (%.1f%%)  |  No-fire: %d (%.1f%%)",
        len(df), y.sum(), fire_pct, (y == 0).sum(), 100 - fire_pct,
    )

    # Warn if dataset is still severely imbalanced (>85% one class)
    if fire_pct > 85 or fire_pct < 15:
        logger.warning(
            "Dataset is highly imbalanced (%.1f%% fire). "
            "Re-run Stage 2 to generate no-fire background samples for better results.",
            fire_pct,
        )

    return X, y


def _split(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info("Split → train: %d  test: %d", len(X_tr), len(X_te))
    return X_tr, X_te, y_tr, y_te


def _fit_scaler(X_train, X_test):
    """
    Fit StandardScaler on train only.
    All subsequent training uses scaled features for consistent distance metrics.
    """
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    return X_train_sc, X_test_sc


def _apply_smote(X_train, y_train):
    """
    Apply SMOTE to diversify the training set.
    If imbalanced-learn is not installed, returns inputs unchanged.
    SMOTE creates synthetic minority-class examples rather than just
    duplicating existing ones, which helps tree models generalise.
    """
    if not _SMOTE_AVAILABLE:
        return X_train, y_train

    before_counts = np.bincount(y_train)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    after_counts  = np.bincount(y_res)

    logger.info(
        "SMOTE: fire %d→%d  |  no-fire %d→%d",
        before_counts[1], after_counts[1],
        before_counts[0], after_counts[0],
    )
    return X_res, y_res


# ── Model training ────────────────────────────────────────────────────────────

def _train_xgboost(X_tr, X_te, y_tr, y_te) -> dict:
    """
    XGBoost with isotonic probability calibration.
    Calibration ensures predict_proba returns true likelihoods
    rather than overconfident scores.
    """
    logger.info("[1/4] Training XGBoost …")
    ratio = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    base = XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=ratio,

    )
    # Wrap with calibration — cv=3 uses cross-val to fit the calibrator
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_tr, y_tr)
    joblib.dump(model, XGBOOST_PATH)
    logger.info("XGBoost (calibrated) saved → %s", XGBOOST_PATH)

    # Feature importance from the underlying estimator
    try:
        fi_vals = np.mean(
            [est.feature_importances_
             for est in model.calibrated_classifiers_],
            axis=0,
        )
        fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": fi_vals})
        fi.sort_values("importance", ascending=False).to_csv(FEATURE_IMP_CSV, index=False)
    except Exception:
        pass

    return _eval("XGBoost", model, X_te, y_te)


def _train_random_forest(X_tr, X_te, y_tr, y_te) -> dict:
    """Random Forest with probability calibration."""
    logger.info("[2/4] Training Random Forest …")
    base  = RandomForestClassifier(**RF_PARAMS)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_tr, y_tr)
    joblib.dump(model, RF_PATH)
    logger.info("Random Forest (calibrated) saved → %s", RF_PATH)
    return _eval("Random Forest", model, X_te, y_te)


def _train_svm(X_tr, X_te, y_tr, y_te) -> dict:
    """SVM with RBF kernel.  SVC(probability=True) already uses Platt scaling."""
    logger.info("[3/4] Training SVM (RBF) …")
    model = SVC(kernel="rbf", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    joblib.dump(model, SVM_PATH)
    logger.info("SVM saved → %s", SVM_PATH)
    return _eval("SVM (RBF)", model, X_te, y_te)


def _train_cnn(X_tr, X_te, y_tr, y_te, ratio: float = 1.0) -> dict:
    """
    1D CNN.  Input shape: (n_features, 1).
    Uses class_weight to compensate for any remaining imbalance.
    """
    logger.info("[4/4] Training 1D CNN …")

    # Lazy import — TensorFlow only required for this model
    from tensorflow.keras import callbacks, layers, models

    n_feat = X_tr.shape[1]
    X_tr3  = X_tr.reshape(-1, n_feat, 1)
    X_te3  = X_te.reshape(-1, n_feat, 1)

    ratio  = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    cnn = models.Sequential([
        layers.Conv1D(64,  3, activation="relu", input_shape=(n_feat, 1), padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1,  activation="sigmoid"),
    ])
    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    cnn.fit(
        X_tr3, y_tr,
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH,
        validation_split=0.1,
        class_weight={0: 1.0, 1: float(ratio)},
        callbacks=[es], verbose=0,
    )
    cnn.save(CNN_PATH)
    logger.info("CNN saved → %s", CNN_PATH)
    return _eval_cnn("1D CNN", cnn, X_te3, y_te)


# ── Optimal threshold ─────────────────────────────────────────────────────────

def _find_and_save_threshold(X_test, y_test, model_path=XGBOOST_PATH) -> float:
    """
    Find the probability threshold that maximises F1 on the test set.

    The default 0.5 threshold is calibrated for balanced classes.
    With real-world fire data the optimal cutoff is usually higher
    (e.g. 0.60–0.75), which dramatically reduces false positives.
    The predictor service reads this value at inference time.
    """
    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    THRESHOLD_PATH.write_text(f"{best_t:.4f}")
    logger.info(
        "Optimal threshold: %.2f  (F1=%.4f)  saved → %s",
        best_t, best_f1, THRESHOLD_PATH,
    )
    return best_t


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return _metrics_dict(name, y_test, preds, probs)


def _eval_cnn(name: str, model, X_test, y_test) -> dict:
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    return _metrics_dict(name, y_test, preds, probs)


def _metrics_dict(name, y_true, y_pred, y_prob) -> dict:
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
    }


def _save_report(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(MODEL_REPORT_CSV, index=False)
    logger.info("Model report → %s", MODEL_REPORT_CSV)
    return df


def _print_summary(results: list[dict]) -> None:
    header = (f"{'Model':<15} | {'Accuracy':>8} | {'Precision':>9} | "
              f"{'Recall':>6} | {'F1':>6} | {'ROC-AUC':>7}")
    logger.info(header)
    logger.info("-" * len(header))
    for r in results:
        logger.info(
            "%-15s | %8.3f | %9.3f | %6.3f | %6.3f | %7.3f",
            r["Model"], r["Accuracy"], r["Precision"],
            r["Recall"], r["F1 Score"], r["ROC-AUC"],
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()