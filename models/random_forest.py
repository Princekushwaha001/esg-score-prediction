# import numpy as np
# from typing import List
#
#
# class DecisionTree:
#     """Simple decision tree for regression."""
#
#     def __init__(self, max_depth: int = 5, min_samples_split: int = 5):
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.tree = None
#
#     def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
#         """Recursively build decision tree."""
#         n_samples, n_features = X.shape
#
#         # Stopping criteria
#         if depth >= self.max_depth or n_samples < self.min_samples_split:
#             return {'value': np.mean(y)}
#
#         # Find best split
#         best_feature, best_threshold = self._find_best_split(X, y)
#
#         if best_feature is None:
#             return {'value': np.mean(y)}
#
#         # Split data
#         left_mask = X[:, best_feature] <= best_threshold
#         right_mask = ~left_mask
#
#         # Build subtrees
#         left_tree = self.fit(X[left_mask], y[left_mask], depth + 1)
#         right_tree = self.fit(X[right_mask], y[right_mask], depth + 1)
#
#         return {
#             'feature': best_feature,
#             'threshold': best_threshold,
#             'left': left_tree,
#             'right': right_tree
#         }
#
#     def _find_best_split(self, X: np.ndarray, y: np.ndarray):
#         """Find best feature and threshold to split on."""
#         best_mse = float('inf')
#         best_feature = None
#         best_threshold = None
#
#         n_features = X.shape[1]
#
#         for feature in range(n_features):
#             thresholds = np.unique(X[:, feature])
#
#             for threshold in thresholds:
#                 left_mask = X[:, feature] <= threshold
#                 right_mask = ~left_mask
#
#                 if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
#                     continue
#
#                 left_y = y[left_mask]
#                 right_y = y[right_mask]
#
#                 mse = self._calculate_mse(left_y, right_y)
#
#                 if mse < best_mse:
#                     best_mse = mse
#                     best_feature = feature
#                     best_threshold = threshold
#
#         return best_feature, best_threshold
#
#     def _calculate_mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
#         """Calculate weighted MSE for a split."""
#         left_mse = np.var(left_y) * len(left_y)
#         right_mse = np.var(right_y) * len(right_y)
#         return (left_mse + right_mse) / (len(left_y) + len(right_y))
#
#     def predict_single(self, x: np.ndarray, tree: dict) -> float:
#         """Predict single sample."""
#         if 'value' in tree:
#             return tree['value']
#
#         if x[tree['feature']] <= tree['threshold']:
#             return self.predict_single(x, tree['left'])
#         else:
#             return self.predict_single(x, tree['right'])
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Predict multiple samples."""
#         return np.array([self.predict_single(x, self.tree) for x in X])
#
#
# class RandomForestModel:
#     """Random Forest for regression."""
#
#     def __init__(self, n_estimators: int = 10, max_depth: int = 5,
#                  min_samples_split: int = 5, random_state: int = 42):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.random_state = random_state
#         self.trees: List[DecisionTree] = []
#
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         """Train random forest with bootstrap sampling."""
#         np.random.seed(self.random_state)
#         n_samples = X.shape[0]
#
#         for i in range(self.n_estimators):
#             # Bootstrap sampling
#             indices = np.random.choice(n_samples, n_samples, replace=True)
#             X_bootstrap = X[indices]
#             y_bootstrap = y[indices]
#
#             # Train tree
#             tree = DecisionTree(max_depth=self.max_depth,
#                               min_samples_split=self.min_samples_split)
#             tree.tree = tree.fit(X_bootstrap, y_bootstrap)
#             self.trees.append(tree)
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """Average predictions from all trees."""
#         predictions = np.array([tree.predict(X) for tree in self.trees])
#         return np.mean(predictions, axis=0)


import numpy as np
from typing import List, Optional, Dict, Tuple
import pickle


class DecisionTree:
    """Regression Decision Tree with proper serialization support."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 5,
                 min_samples_leaf: int = 2, random_state: Optional[int] = None):
        """Initialize Decision Tree."""
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Recursively build decision tree."""
        n_samples, n_features = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'value': float(np.mean(y)), 'n_samples': n_samples}

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return {'value': float(np.mean(y)), 'n_samples': n_samples}

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check minimum samples in leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return {'value': float(np.mean(y)), 'n_samples': n_samples}

        # Build subtrees
        left_tree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.fit(X[right_mask], y[right_mask], depth + 1)

        self.tree = {
            'feature': best_feature,
            'threshold': float(best_threshold),
            'left': left_tree,
            'right': right_tree,
            'n_samples': n_samples,
            'value': float(np.mean(y))
        }

        return self.tree

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find best feature and threshold to split on."""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                mse = self._calculate_mse(left_y, right_y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _calculate_mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate weighted MSE for a split."""
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        if n_total == 0:
            return float('inf')

        var_left = np.var(left_y) if n_left > 0 else 0
        var_right = np.var(right_y) if n_right > 0 else 0

        weighted_var = (n_left / n_total) * var_left + (n_right / n_total) * var_right
        return float(weighted_var)

    def predict_single(self, x: np.ndarray, tree: Optional[Dict] = None) -> float:
        """Predict single sample."""
        if tree is None:
            tree = self.tree

        if 'value' in tree and tree.get('feature') is None:
            return float(tree['value'])

        feature = tree.get('feature')
        threshold = tree.get('threshold')

        if feature is None:
            return float(tree.get('value', 0.0))

        if x[feature] <= threshold:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multiple samples."""
        if self.tree is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        return np.array([self.predict_single(x, self.tree) for x in X])

    def get_params(self) -> Dict:
        """Return model parameters for serialization."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'tree': self.tree
        }

    def set_params(self, params: Dict):
        """Set model parameters for deserialization."""
        self.max_depth = params['max_depth']
        self.min_samples_split = params['min_samples_split']
        self.min_samples_leaf = params['min_samples_leaf']
        self.random_state = params['random_state']
        self.tree = params['tree']


class RandomForestModel:
    """Random Forest with proper serialization support."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 15,
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 max_features: Optional[str] = None, random_state: int = 42):
        """Initialize Random Forest."""
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_data: List = []  # Store tree data, not objects
        self.feature_importance = None

    def _get_n_features_to_sample(self, n_features: int) -> int:
        """Determine number of features to sample."""
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError("max_features must be 'sqrt', 'log2', or None")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train random forest with bootstrap and feature sampling."""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_features_sample = self._get_n_features_to_sample(n_features)

        feature_importances = np.zeros(n_features)
        self.trees_data = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Random feature sampling
            feature_indices = np.random.choice(n_features, n_features_sample, replace=False)
            X_sampled = X_bootstrap[:, feature_indices]

            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            tree.fit(X_sampled, y_bootstrap)

            # Store tree data instead of object
            self.trees_data.append({
                'tree': tree.tree,
                'feature_indices': feature_indices.tolist()
            })

            # Track feature importance
            for idx in feature_indices:
                feature_importances[idx] += 1

        self.feature_importance = feature_importances / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all trees."""
        if not self.trees_data:
            raise ValueError("Model not trained yet! Call fit() first.")

        predictions = []

        for tree_data in self.trees_data:
            tree_obj = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree_obj.tree = tree_data['tree']
            feature_indices = tree_data['feature_indices']

            X_sampled = X[:, feature_indices]
            pred = tree_obj.predict(X_sampled)
            predictions.append(pred)

        return np.mean(predictions, axis=0)

    def get_feature_importance(self) -> Dict:
        """Get feature importance scores."""
        return {
            'importances': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features': len(self.feature_importance) if self.feature_importance is not None else 0
        }

    def get_params(self) -> Dict:
        """Return model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'trees_data': self.trees_data,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
