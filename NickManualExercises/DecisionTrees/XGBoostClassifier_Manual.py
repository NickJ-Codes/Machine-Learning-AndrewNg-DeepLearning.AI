import time

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from XGBoostClassifier_Guided import clean_data, replace_missing_data, splitXy
import pandas as pd

import numpy as np
from typing import Dict, Union, Tuple

class SimpleXGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, min_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.tree = None

    def find_best_split(self, X: np.ndarray, residuals: np.ndarray, feature_idx: int) -> Tuple[float, float]:
        """Highly optimized split finder"""
        feature_values = X[:, feature_idx]
        n_samples = len(residuals)

        if n_samples < 2 * self.min_samples_split:
            return None, -np.inf

        # Pre-calculate total sum for efficiency
        total_sum = np.sum(residuals)
        total_count = n_samples

        # Find best split using vectorized operations
        best_gain = -np.inf
        best_threshold = None

        # Calculate cumulative sums for efficient splitting
        sorted_idx = np.argsort(feature_values)
        sorted_residuals = residuals[sorted_idx]
        cumsum = np.cumsum(sorted_residuals)

        # Try different bin edges as split points
        for i in range(self.min_samples_split, n_samples - self.min_samples_split):
            if feature_values[sorted_idx[i]] == feature_values[sorted_idx[i - 1]]:
                continue

            left_sum = cumsum[i]
            right_sum = total_sum - left_sum

            left_count = i + 1
            right_count = total_count - left_count

            # Fast gain calculation
            gain = (left_sum * left_sum / left_count +
                    right_sum * right_sum / right_count) / (total_sum * total_sum / total_count + 1e-10)

            if gain > best_gain:
                best_gain = gain
                best_threshold = (feature_values[sorted_idx[i]] + feature_values[sorted_idx[i - 1]]) / 2

        return best_threshold, best_gain

    def build_tree(self, X: np.ndarray, residuals: np.ndarray, depth: int = 0) -> Union[Dict, float]:
        """Optimized tree building"""
        n_samples = len(residuals)

        # Early stopping conditions
        if (depth >= self.max_depth or
                n_samples < 2 * self.min_samples_split or
                np.allclose(residuals, 0, rtol=1e-7)):
            return np.sum(residuals) / (n_samples + 1)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        # Consider subset of features for large datasets
        n_features = X.shape[1]
        n_features_to_consider = min(n_features, max(int(np.sqrt(n_features)), 5))
        feature_indices = np.random.choice(n_features, n_features_to_consider, replace=False)

        # Find best split
        for feature in feature_indices:
            threshold, gain = self.find_best_split(X, residuals, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        if best_gain <= self.min_gain:
            return np.sum(residuals) / (n_samples + 1)

        # Use boolean indexing for splitting
        mask = X[:, best_feature] <= best_threshold

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.build_tree(X[mask], residuals[mask], depth + 1),
            'right': self.build_tree(X[~mask], residuals[~mask], depth + 1)
        }

class SimpleXGBoost:
    def __init__(self, n_estimators=3, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_gain=0.0, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.subsample = subsample
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleXGBoost':
        """Optimized training process"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        # Pre-allocate memory for predictions
        sample_size = int(n_samples * self.subsample)

        for _ in range(self.n_estimators):
            print(f"training tree #{_}")
            # Calculate residuals
            residuals = y - 1.0 / (1.0 + np.exp(-predictions))

            # Subsample data
            if self.subsample < 1.0:
                idx = np.random.choice(n_samples, sample_size, replace=False)
                X_sample, residuals_sample = X[idx], residuals[idx]
            else:
                X_sample, residuals_sample = X, residuals

            # Create and train new tree
            tree = SimpleXGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain
            )

            tree.tree = tree.build_tree(X_sample, residuals_sample)
            self.trees.append(tree)

            # Update predictions
            predictions += self.learning_rate * self._predict_batch(X, tree.tree)

        return self

    def _predict_batch(self, X: np.ndarray, tree: Union[Dict, float]) -> np.ndarray:
        """Optimized batch prediction"""
        if isinstance(tree, float):
            return np.full(X.shape[0], tree)

        predictions = np.zeros(X.shape[0])
        mask = X[:, tree['feature']] <= tree['threshold']

        predictions[mask] = self._predict_batch(X[mask], tree['left'])
        predictions[~mask] = self._predict_batch(X[~mask], tree['right'])
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Optimized probability prediction"""
        scores = sum(self.learning_rate * self._predict_batch(X, tree.tree)
                     for tree in self.trees)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int8)

def main():
    df_raw = pd.read_csv('../data/Telco_customer_churn.csv')
    df = clean_data(df_raw,verbose = False)
    df = replace_missing_data(df, verbose = False)
    # print(df.dtypes)
    X, y = splitXy(df)

    X_encoded = pd.get_dummies(X, columns = ['City',
                                             'Gender',
                                             'Senior_Citizen',
                                             'Partner',
                                             'Dependents',
                                             'Phone_Service',
                                             'Multiple_Lines',
                                             "Internet_Service",
                                             'Online_Security',
                                             "Online_Backup",
                                             'Device_Protection',
                                             'Tech_Support',
                                             'Streaming_TV',
                                             'Streaming_Movies',
                                             'Contract',
                                             'Paperless_Billing',
                                             'Payment_Method'])

    X_encoded = X_encoded.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = SimpleXGBoost(n_estimators=100, learning_rate=0.1, max_depth=3)
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create and plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did not leave", "Left"]
    )

    # Plot the confusion matrix
    disp.plot(cmap='Blues', values_format='d')

    # Add title and display the plot
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
