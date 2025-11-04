import numpy as np
from typing import Dict


class ModelEvaluator:
    """Calculate evaluation metrics for regression models."""

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² (coefficient of determination) score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    @classmethod
    def evaluate_all(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics."""
        return {
            'r2_score': cls.r2_score(y_true, y_pred),
            'mae': cls.mae(y_true, y_pred),
            'rmse': cls.rmse(y_true, y_pred),
            'mse': cls.mse(y_true, y_pred)
        }
