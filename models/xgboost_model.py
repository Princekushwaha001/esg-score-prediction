# from xgboost import XGBRegressor
# import numpy as np
#
#
# class XGBoostModel:
#     """XGBoost wrapper for ESG score prediction."""
#
#     def __init__(self, n_estimators: int = 100, max_depth: int = 6,
#                  learning_rate: float = 0.1, random_state: int = 42):
#         self.model = XGBRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=random_state,
#             objective='reg:squarederror'
#         )
#
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """Train XGBoost model."""
#         self.model.fit(X, y)
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Make predictions."""
#         return self.model.predict(X)
#
#     def get_feature_importance(self) -> dict:
#         """Get feature importance scores."""
#         return dict(zip(range(len(self.model.feature_importances_)),
#                         self.model.feature_importances_))


from xgboost import XGBRegressor
import numpy as np
from typing import Dict, Optional


class XGBoostModel:
    """XGBoost wrapper with configurable hyperparameters and validation."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42,
                 early_stopping_rounds: Optional[int] = None):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds (None to disable)
        """
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("learning_rate must be between 0 and 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective='reg:squarederror',
            eval_metric='rmse',
            verbosity=0
        )

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[tuple] = None):
        """
        Train XGBoost model.

        Args:
            X: Training features
            y: Training targets
            eval_set: Evaluation set for early stopping
        """
        fit_params = {'X': X, 'y': y}

        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['eval_set'] = [eval_set]
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds

        self.model.fit(**fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict:
        """Get feature importance scores with interpretation."""
        importances = self.model.feature_importances_
        return {
            'importances': importances.tolist(),
            'n_features': len(importances),
            'top_features': np.argsort(importances)[::-1].tolist()
        }

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state
        }
