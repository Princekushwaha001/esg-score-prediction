import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataPreprocessor:
    """Preprocess ESG data for model training."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns = ['CO2_Emissions', 'Energy_Use', 'Diversity_Index', 'Governance_Rating']
        self.target_column = 'ESG_Score'

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from dataframe.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (X, y) arrays
        """
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        return X, y

    def train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train and test sets.

        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state)

    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """
        Normalize features using training set statistics.

        Returns:
            Normalized X_train, X_test
        """
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        X_train_norm = (X_train - mean) / (std + 1e-8)
        X_test_norm = (X_test - mean) / (std + 1e-8)

        return X_train_norm, X_test_norm
