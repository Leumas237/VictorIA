"""
lightgbm_model.py – LightGBM multi-class classifier for match prediction.
Classes: 0=HomeWin, 1=Draw, 2=AwayWin
"""
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config
from models.base_model import BaseModel

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LGBM_AVAILABLE = False


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self):
        self.cv_score: Optional[float] = None
        self._trained = False
        if not LGBM_AVAILABLE:
            self.model = None
            return
        self.model = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=-1,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if not LGBM_AVAILABLE or self.model is None:
            print("[LightGBM] lightgbm package not installed – model disabled.")
            return
        cv = StratifiedKFold(n_splits=config.STRATIFIED_FOLDS, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="accuracy", n_jobs=1
        )
        self.cv_score = float(scores.mean())
        print(f"[LightGBM] CV Accuracy: {self.cv_score:.3f} ± {scores.std():.3f}")
        self.model.fit(X, y)
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained or self.model is None:
            raise RuntimeError("LightGBM model not trained.")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model is None:
            return np.zeros(0, dtype=float)
        return self.model.feature_importances_
