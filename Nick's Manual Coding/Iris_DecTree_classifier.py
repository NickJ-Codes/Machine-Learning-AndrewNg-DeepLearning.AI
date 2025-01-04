from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target

# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# create and train the DT classifer
dt_classifier = DecisionTreeClassifier(random_state = 42)
dt_classifier.fit(X_train, y_train)

# make predictions
y_pred = dt_classifier.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.2f}")

# Display detailed classification report
print("\n Classification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt_classifier,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               filled=True,
               rounded=True,
               fontsize=10,  # Add font size for better readability
               proportion=True,  # Show proportions at each node
               precision=2)  # Set decimal precision for numbers
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')  # Save high-quality image
plt.show()

# Optional: Clear the plot to free memory
plt.close()
