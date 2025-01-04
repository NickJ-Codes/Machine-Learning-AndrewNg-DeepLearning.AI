import numpy as np
from collections import Counter
from sklearn.datasets import load_iris

"""
The code in this file is pretty much identical to to the v1 file
In the v1 file, i manually typed (but did not copy-paste) the code from the genAI tools.
In the v2 file, i copied the skeleton (class, class functions, etc.) but wrote
the code within the functions and class defintions myself, although had to debug using the correct V1 version of the code 
"""

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None, depth = 0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth

class DecisionTreeClassifier:
    def __init__(self, max_depth = None, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.tree_depth = 0

    def get_depth(self):
        """Returns the total depth of the tree"""
        return self.tree_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # update tree depth
        self.tree_depth = max(self.tree_depth, depth)

        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split \
                or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        #find the best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None: # no valid split found
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)

        # create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:,feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        #generate split
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n)*e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist/len(y) #converts bin counts to frequencies
        entropy = -np.sum([p*np.log2(p) for p in ps if p>0])
        return entropy

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        # Counter returns a dictionary like object with key:value pairs sorted
        # [1] returns a list of tuples, which all have the most count
        # [0] selects the first tuple in the dictionary (key value pair)
        # [0] again selects the key of the key/value pair ([1] would select the value)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None: # Node values are not none at the final leafs
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    #split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    #train the classifier
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X_train, y_train)

    # make predictions
    predictions = clf.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Show accuracy as a function of depth:
    for x in range(10):
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        tree_depth = clf.tree_depth
        print(f"Tree depth: {tree_depth}, Accuracy: {accuracy:.2f}")


# testing the implementation
if __name__ == "__main__":
    main()
