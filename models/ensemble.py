"""
ensemble.py – Weighted soft-voting ensemble combining XGBoost, RF, LightGBM and NN.
Automatically trains all sub-models if not already cached.
"""
from typing import List, Optional

import numpy as np
import joblib

import config
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from models.lightgbm_model import LightGBMModel, LGBM_AVAILABLE
from models.neural_net import NeuralNetModel, TF_AVAILABLE

LABELS = ["Victoire Domicile", "Nul", "Victoire Extérieur"]
LABEL_SHORT = ["HomeWin", "Draw", "AwayWin"]


class EnsembleModel:
    """
    Weighted soft-voting ensemble.
    Exposes confidence scores and individual model breakdowns.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = list(weights or config.ENSEMBLE_WEIGHTS)
        self.xgb = XGBoostModel()
        self.rf = RandomForestModel()
        self.lgbm = LightGBMModel()
        self.nn = NeuralNetModel()
        self._trained = False

    def _lgbm_active(self) -> bool:
        return (
            LGBM_AVAILABLE
            and self.lgbm._trained
            and self.weights[config.ENSEMBLE_IDX_LGBM] > 0
        )

    def _nn_active(self) -> bool:
        return (
            TF_AVAILABLE
            and self.nn._trained
            and self.nn.model is not None
            and self.weights[config.ENSEMBLE_IDX_NN] > 0
        )

    def active_model_names(self) -> List[str]:
        names = ["XGBoost", "RandomForest"]
        if self._lgbm_active():
            names.append("LightGBM")
        if self._nn_active():
            names.append("NeuralNet")
        return names

    def _effective_weights(self) -> np.ndarray:
        w = np.array(self.weights, dtype=float)
        if not self._lgbm_active():
            w[config.ENSEMBLE_IDX_LGBM] = 0.0
        if not self._nn_active():
            w[config.ENSEMBLE_IDX_NN] = 0.0
        total = w.sum()
        if total <= 0:
            return np.array(config.ENSEMBLE_FALLBACK_WEIGHTS, dtype=float)
        return w / total

    # ──────────────────────────────────────────────────────────
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print("[Ensemble] Training sub-models …")
        self.xgb.train(X, y)
        self.rf.train(X, y)
        if self.weights[config.ENSEMBLE_IDX_LGBM] > 0 and LGBM_AVAILABLE:
            self.lgbm.train(X, y)
        else:
            print("[Ensemble] LightGBM skipped (disabled or not installed).")
        if self.weights[config.ENSEMBLE_IDX_NN] > 0 and TF_AVAILABLE:
            self.nn.train(X, y)
        else:
            print("[Ensemble] NeuralNet skipped (disabled or TensorFlow absent).")
        self._adapt_weights_from_cv()
        self._trained = True
        self._save()
        print("[Ensemble] All sub-models trained and saved.")

    def _adapt_weights_from_cv(self) -> None:
        cv = {
            "XGBoost": self.xgb.cv_score or 0.0,
            "RandomForest": self.rf.cv_score or 0.0,
            "LightGBM": self.lgbm.cv_score or 0.0,
            "NeuralNet": self.nn.cv_score or 0.0,
        }
        names = ["XGBoost", "RandomForest", "LightGBM", "NeuralNet"]
        enabled = [
            True,
            True,
            self.weights[config.ENSEMBLE_IDX_LGBM] > 0 and self.lgbm.cv_score is not None,
            self.weights[config.ENSEMBLE_IDX_NN] > 0 and self.nn.cv_score is not None,
        ]
        base = np.array(self.weights, dtype=float)
        perf = np.array([max(cv[n], 0.01) for n in names], dtype=float)
        raw = base * perf
        raw[~np.array(enabled)] = 0.0
        total = raw.sum()
        if total > 0:
            self.weights = (raw / total).tolist()

    def _save(self):
        joblib.dump(self, config.ENSEMBLE_PATH)

    @classmethod
    def load(cls) -> "EnsembleModel":
        if not config.ENSEMBLE_PATH.exists():
            raise FileNotFoundError(
                f"Modèles introuvables. Exécutez : python train.py"
            )
        print("[Ensemble] Loading from cache …")
        return joblib.load(config.ENSEMBLE_PATH)

    @classmethod
    def load_or_train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        force_retrain: bool = False,
    ) -> "EnsembleModel":
        if config.ENSEMBLE_PATH.exists() and not force_retrain:
            return cls.load()
        obj = cls()
        obj.train(X, y)
        return obj

    # ──────────────────────────────────────────────────────────
    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, 3) probability matrix."""
        p_xgb = self.xgb.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        w = self._effective_weights()
        combined = (
            w[config.ENSEMBLE_IDX_XGB] * p_xgb +
            w[config.ENSEMBLE_IDX_RF] * p_rf
        )
        if self._lgbm_active():
            p_lgbm = self.lgbm.predict_proba(X)
            combined = combined + w[config.ENSEMBLE_IDX_LGBM] * p_lgbm
        if self._nn_active():
            p_nn = self.nn.predict_proba(X)
            combined = combined + w[config.ENSEMBLE_IDX_NN] * p_nn
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return combined / row_sums

    def predict(self, X: np.ndarray) -> dict:
        """
        Returns a rich prediction dict for a single sample (or batch of 1).
        """
        combined_batch = self.predict_proba_batch(X)
        combined = combined_batch[0]

        p_xgb = self.xgb.predict_proba(X)[0]
        p_rf = self.rf.predict_proba(X)[0]
        p_lgbm = (
            self.lgbm.predict_proba(X)[0]
            if self._lgbm_active()
            else np.full(3, 1 / 3)
        )
        p_nn = (
            self.nn.predict_proba(X)[0]
            if self._nn_active()
            else np.full(3, 1 / 3)
        )

        outcome_idx = int(np.argmax(combined))
        confidence = float(combined[outcome_idx]) * 100

        all_preds = [p_xgb, p_rf]
        if self._lgbm_active():
            all_preds.append(p_lgbm)
        if self._nn_active():
            all_preds.append(p_nn)
        all_preds = np.array(all_preds)
        agreement = self._agreement_score(all_preds)

        model_breakdown = {
            "XGBoost": {
                k: round(v * 100, 1)
                for k, v in zip(LABEL_SHORT, p_xgb)
            },
            "RandomForest": {
                k: round(v * 100, 1)
                for k, v in zip(LABEL_SHORT, p_rf)
            },
        }
        if self._lgbm_active():
            model_breakdown["LightGBM"] = {
                k: round(v * 100, 1)
                for k, v in zip(LABEL_SHORT, p_lgbm)
            }
        if self._nn_active():
            model_breakdown["NeuralNet"] = {
                k: round(v * 100, 1)
                for k, v in zip(LABEL_SHORT, p_nn)
            }

        return {
            "probabilities": {
                "home_win": round(float(combined[0]) * 100, 1),
                "draw": round(float(combined[1]) * 100, 1),
                "away_win": round(float(combined[2]) * 100, 1),
            },
            "outcome": LABELS[outcome_idx],
            "outcome_key": LABEL_SHORT[outcome_idx],
            "confidence": round(confidence, 1),
            "model_breakdown": model_breakdown,
            "agreement": round(agreement, 3),
            "raw_proba": combined,
            "active_models": self.active_model_names(),
        }

    @staticmethod
    def _agreement_score(preds: np.ndarray) -> float:
        """Average cosine similarity between model predictions."""
        from itertools import combinations

        sims = []
        for a, b in combinations(preds, 2):
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            sims.append(float(np.dot(a, b) / norm) if norm > 0 else 0.0)
        return float(np.mean(sims))

    @property
    def cv_scores(self) -> dict:
        return {
            "XGBoost": self.xgb.cv_score,
            "RandomForest": self.rf.cv_score,
            "LightGBM": self.lgbm.cv_score if self._lgbm_active() else None,
            "NeuralNet": self.nn.cv_score if self._nn_active() else None,
        }
