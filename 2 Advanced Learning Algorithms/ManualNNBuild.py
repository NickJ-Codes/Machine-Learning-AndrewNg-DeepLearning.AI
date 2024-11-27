# Using the Iris data set, I will manually build a neural network, without using TF

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# pre process the data
scaler = StandardScaler()
x = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Convert target labels to one-hot encoding
def one_hot_encoding(y):
    n_classes = len(np.unique(y))
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1 #smart, np.arange locations position x, the values of
    return one_hot

y_train_encoded = one_hot_encoding(y_train)
y_test_encoded = one_hot_encoding(y_test)

# Define the neurla network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1,hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.b2 = np.zeros((1, output_size))
        self.accuracy_history = []

    def calculate_accuracy(self, X, y):
        predictions = self.forward_pass(X)
        predicted_classes = np.argmax(predictions, axis = 1)
        true_classes = np.argmax(y, axis = 1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = softmax(self.z2)
        return self.y_hat

    def backward_pass(self, X, y, learning_rate):
        batch_size = X.shape[0]

        # Calculate gradients using stored values
        delta2 = self.y_hat - y
        dW2 = np.dot(self.a1.T, delta2) / batch_size
        db2 = np.sum(delta2, axis=0, keepdims = True) / batch_size

        # using stored self.a1
        delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1-self.a1)
        dW1 = np.dot(X.T, delta1) / batch_size
        db1 = np.sum(delta1, axis=0, keepdims = True) / batch_size

        # update weights & biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            #forward pass
            y_hat = self.forward_pass(X)

            # Backward pass
            self.backward_pass(X, y, learning_rate)

            #calculate accuracy and store it for this epoch
            accuracy = self.calculate_accuracy(X, y)
            self.accuracy_history.append(accuracy)

            #Print progress every 25 results
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}, Accuracy: {accuracy:.4f}")

# After training, you can plot the accuracy history
def plot_accuracy(model):
    plt.figure(figsize=(10, 6))
    plt.plot(model.accuracy_history)
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Create and train the neural network
model = NeuralNetwork(4, 10, 3)
model.train(X_train, y_train_encoded, epochs = 1000, learning_rate = 0.05)

# Make predictions on the test set
y_pred = model.forward_pass(X_test)
predicted_classes = np.argmax(y_pred, axis = 1)
true_classes = np.argmax(y_test_encoded, axis = 1)
# Calculate accuracy on the test set
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test set Accuracy: {accuracy:.4f}")

plot_accuracy(model)