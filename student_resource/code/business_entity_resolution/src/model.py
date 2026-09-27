"""
Matching Model module using LightGBM GBDT with early stopping.
Optimized for high-precision Macro F0.5 on entity resolution.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from xgboost import XGBClassifier
from .config import MODEL_PARAMS, MODEL_DIR
from .feature_extraction import FEATURE_NAMES


class EntityMatchingModel:
    """
    High-precision GBDT pairwise matching model optimized for Macro F0.5.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or MODEL_PARAMS
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.use_lightgbm = HAS_LIGHTGBM

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the matching model with optional early stopping on validation set.
        """
        if self.use_lightgbm:
            # Use LightGBM native API to avoid sklearn wrapper deprecation warnings
            lgb_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": self.params.get("learning_rate", 0.05),
                "num_leaves": self.params.get("num_leaves", 127),
                "max_depth": self.params.get("max_depth", 8),
                "min_child_samples": self.params.get("min_child_samples", 30),
                "subsample": self.params.get("subsample", 0.85),
                "subsample_freq": 1,
                "colsample_bytree": self.params.get("colsample_bytree", 0.85),
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            n_estimators = self.params.get("n_estimators", 800)

            dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            callbacks = [lgb.log_evaluation(period=-1)]  # silence per-iter output

            if X_val is not None and y_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
                callbacks.append(lgb.early_stopping(stopping_rounds=40, verbose=False))
                self.model = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=n_estimators,
                    valid_sets=[dval],
                    callbacks=callbacks
                )
            else:
                self.model = lgb.train(
                    lgb_params,
                    dtrain,
                    num_boost_round=n_estimators,
                    callbacks=callbacks
                )
            self._use_native_lgb = True
        else:
            self._use_native_lgb = False
            self.model = XGBClassifier(
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 8),
                n_estimators=self.params.get("n_estimators", 800),
                subsample=self.params.get("subsample", 0.85),
                colsample_bytree=self.params.get("colsample_bytree", 0.85),
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                early_stopping_rounds=40 if X_val is not None else None
            )
            eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of matching (P(match=1)).
        """
        if len(X) == 0:
            return np.array([])
        if getattr(self, "_use_native_lgb", False):
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Return dict mapping feature name to importance score.
        """
        if self.model is None:
            return {}
        if getattr(self, "_use_native_lgb", False):
            imps = self.model.feature_importance(importance_type="gain")
        else:
            imps = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, imps)}

    def save(self, filepath: Path) -> None:
        """Serialize model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.model, getattr(self, "_use_native_lgb", False)), filepath)

    def load(self, filepath: Path) -> None:
        """Load model from disk."""
        payload = joblib.load(filepath)
        if isinstance(payload, tuple):
            self.model, self._use_native_lgb = payload
        else:
            # Legacy: sklearn wrapper
            self.model = payload
            self._use_native_lgb = False
