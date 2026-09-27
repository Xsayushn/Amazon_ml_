"""
LightGBM Matching Classifier and F_0.5 Optimizer
Amazon ML Challenge 2026
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np

def compute_macro_f05(gt_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]], all_s1_ids: List[str]) -> float:
    """
    Compute macro-averaged F_0.5 score across all S1 entities, including singletons.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    scores = []
    for s1_id in all_s1_ids:
        y_true = gt_dict.get(s1_id, set())
        y_pred = pred_dict.get(s1_id, set())
        
        # Singleton handling
        if not y_true:
            scores.append(1.0 if not y_pred else 0.0)
            continue
            
        if not y_pred:
            scores.append(0.0)
            continue
            
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if prec + rec == 0.0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)
            
    return float(np.mean(scores))


class EntityMatcher:
    def __init__(self, params: dict = None, num_boost_round: int = 350, threshold: float = 0.55):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.85,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42
        }
        self.num_boost_round = num_boost_round
        self.threshold = threshold
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train LightGBM binary classifier."""
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def save(self, filepath: Path):
        """Save model to disk."""
        if self.model is not None:
            self.model.save_model(str(filepath))

    def load(self, filepath: Path):
        """Load model from disk."""
        self.model = lgb.Booster(model_file=str(filepath))
