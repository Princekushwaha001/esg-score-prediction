# import numpy as np
# from typing import Optional
#
#
# class LinearRegressionModel:
#     """Linear Regression using Normal Equation."""
#
#     def __init__(self):
#         self.coefficients = None
#         self.intercept = None
#
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """
#         Train linear regression model using normal equation.
#
#         Args:
#             X: Training features (n_samples, n_features)
#             y: Training targets (n_samples,)
#         """
#         # Add bias term
#         X_with_bias = np.c_[np.ones(X.shape[0]), X]
#
#         # Normal equation: β = (X'X)^(-1)X'y
#         try:
#             XtX = X_with_bias.T @ X_with_bias
#             Xty = X_with_bias.T @ y
#             theta = np.linalg.solve(XtX, Xty)
#
#             self.intercept = theta[0]
#             self.coefficients = theta[1:]
#         except np.linalg.LinAlgError:
#             # Fallback to pseudoinverse if singular
#             theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
#             self.intercept = theta[0]
#             self.coefficients = theta[1:]
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Make predictions.
#
#         Args:
#             X: Input features (n_samples, n_features)
#
#         Returns:
#             Predictions (n_samples,)
#         """
#         if self.coefficients is None:
#             raise ValueError("Model not trained yet!")
#
#         return self.intercept + X @ self.coefficients
#
#     def get_params(self) -> dict:
#         """Return model parameters."""
#         return {
#             'intercept': self.intercept,
#             'coefficients': self.coefficients.tolist() if self.coefficients is not None else None
#         }


import numpy as np
from typing import Optional, Dict, Tuple


class LinearRegressionModel:
    """Linear Regression using Normal Equation with L2 Regularization (Ridge)."""

    def __init__(self, regularization: float = 0.0, normalize: bool = True):
        """
        Initialize Linear Regression model.

        Args:
            regularization: L2 regularization strength (0 = no regularization)
            normalize: Whether to normalize features
        """
        self.coefficients = None
        self.intercept = None
        self.regularization = regularization
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train linear regression model using normal equation with regularization.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)

        Raises:
            ValueError: If input dimensions are invalid
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")

        if X.shape[0] < X.shape[1]:
            raise ValueError(f"Number of samples ({X.shape[0]}) must be >= number of features ({X.shape[1]})")

        # Normalize features if specified
        if self.normalize:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
            X_normalized = (X - self.feature_mean) / self.feature_std
        else:
            X_normalized = X

        # Add bias term
        X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

        # Ridge regression: β = (X'X + λI)^(-1)X'y
        try:
            XtX = X_with_bias.T @ X_with_bias

            # Add regularization term
            if self.regularization > 0:
                XtX[1:, 1:] += self.regularization * np.eye(X_with_bias.shape[1] - 1)

            Xty = X_with_bias.T @ y
            theta = np.linalg.solve(XtX, Xty)

            self.intercept = theta[0]
            self.coefficients = theta[1:]

        except np.linalg.LinAlgError as e:
            # Fallback to pseudoinverse if singular
            print(f"Warning: Matrix is singular, using pseudoinverse. Error: {e}")
            theta = np.linalg.pinv(X_with_bias) @ y
            self.intercept = theta[0]
            self.coefficients = theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            ValueError: If model not trained or input shape is invalid
        """
        if self.coefficients is None:
            raise ValueError("Model not trained yet! Call fit() first.")

        if X.shape[1] != self.coefficients.shape[0]:
            raise ValueError(f"Expected {self.coefficients.shape[0]} features, got {X.shape[1]}")

        # Normalize if needed
        if self.normalize:
            X = (X - self.feature_mean) / self.feature_std

        return self.intercept + X @ self.coefficients

    def get_params(self) -> Dict:
        """Return model parameters."""
        return {
            'intercept': float(self.intercept) if self.intercept is not None else None,
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else None,
            'regularization': self.regularization,
            'normalize': self.normalize
        }

    def get_r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
