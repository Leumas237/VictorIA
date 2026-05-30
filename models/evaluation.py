"""
evaluation.py – Hold-out metrics for the ensemble (accuracy, log loss, confusion matrix).
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split

import config


def baseline_accuracy(y: np.ndarray) -> float:
    """Always predict the majority class."""
    values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def evaluate_ensemble_chronological(
    ensemble,
    X: np.ndarray,
    y: np.ndarray,
    sample_seasons: List[int],
    test_season: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate with temporal split: train on older seasons, test on latest season.
    """
    seasons_arr = np.array(sample_seasons)
    if test_season is None:
        test_season = int(max(sample_seasons))

    train_mask = seasons_arr != test_season
    test_mask = seasons_arr == test_season

    if test_mask.sum() < 10 or train_mask.sum() < 50:
        # Fallback: last 20% chronologically
        split_idx = int(len(X) * 0.8)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split_idx] = True
        test_mask = ~train_mask
        test_season = int(max(sample_seasons))
        eval_mode = "chronological_80_20"
    else:
        eval_mode = f"season_holdout_{test_season}"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    ensemble.train(X_train, y_train)

    proba = ensemble.predict_proba_batch(X_test)
    y_pred = np.argmax(proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    base = baseline_accuracy(y_test)

    train_seasons = sorted(set(seasons_arr[train_mask].tolist()))
    test_seasons = sorted(set(seasons_arr[test_mask].tolist()))

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "baseline_accuracy": round(base, 4),
        "improvement_vs_baseline": round(acc - base, 4),
        "confusion_matrix": cm,
        "class_labels": ["HomeWin", "Draw", "AwayWin"],
        "per_class_metrics": _per_class_metrics(y_test, y_pred),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_scores": ensemble.cv_scores,
        "active_models": ensemble.active_model_names(),
        "validation": _validation_metrics(ensemble, X_train, y_train),
        "eval_mode": eval_mode,
        "test_season": test_season,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
    }


def evaluate_ensemble(
    ensemble,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate ensemble on a stratified hold-out split.
    Expects ensemble already trained on full data or train split only.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Refit on train only for unbiased hold-out metrics
    ensemble.train(X_train, y_train)

    proba = ensemble.predict_proba_batch(X_test)
    y_pred = np.argmax(proba, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    base = baseline_accuracy(y_test)

    return {
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "baseline_accuracy": round(base, 4),
        "improvement_vs_baseline": round(acc - base, 4),
        "confusion_matrix": cm,
        "class_labels": ["HomeWin", "Draw", "AwayWin"],
        "per_class_metrics": _per_class_metrics(y_test, y_pred),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_scores": ensemble.cv_scores,
        "active_models": ensemble.active_model_names(),
        "validation": _validation_metrics(ensemble, X_train, y_train),
    }


def load_metrics(path) -> Optional[Dict[str, Any]]:
    import json

    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    names = ["HomeWin", "Draw", "AwayWin"]
    return {
        names[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(names))
    }


def _validation_metrics(ensemble, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Extra robust validation from train split only:
    - Time series split on chronological sample order.
    - Stratified k-fold for balanced class evaluation.
    Uses XGBoost as a stable proxy estimator to avoid nested re-training costs
    of the full ensemble inside each validation fold.
    """
    metrics: Dict[str, Any] = {}
    if len(X_train) > config.TIME_SERIES_SPLITS + 5:
        tscv = TimeSeriesSplit(
            n_splits=min(
                config.TIME_SERIES_SPLITS,
                max(len(X_train) // config.MIN_SAMPLES_PER_TIMESERIES_FOLD, 2),
            )
        )
        try:
            ts_scores = cross_val_score(
                ensemble.xgb.model,
                X_train,
                y_train,
                cv=tscv,
                scoring="accuracy",
                n_jobs=1,
            )
            metrics["time_series_cv_accuracy_mean"] = round(float(np.mean(ts_scores)), 4)
            metrics["time_series_cv_accuracy_std"] = round(float(np.std(ts_scores)), 4)
        except ValueError as exc:
            print(f"[Evaluation] Time-series CV skipped: {exc}")
    class_counts = np.bincount(y_train)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    if min_class_count >= 2:
        skf = StratifiedKFold(
            n_splits=min(config.STRATIFIED_FOLDS, min_class_count),
            shuffle=True,
            random_state=42,
        )
        try:
            skf_scores = cross_val_score(
                ensemble.xgb.model,
                X_train,
                y_train,
                cv=skf,
                scoring="accuracy",
                n_jobs=1,
            )
            metrics["stratified_cv_accuracy_mean"] = round(float(np.mean(skf_scores)), 4)
            metrics["stratified_cv_accuracy_std"] = round(float(np.std(skf_scores)), 4)
        except ValueError as exc:
            print(f"[Evaluation] Stratified CV skipped: {exc}")
    return metrics
