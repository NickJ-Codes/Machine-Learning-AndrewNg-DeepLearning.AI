# Load Iris Data set
# use a 2 layer (10, 5) neural network to classify iris dataset

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib
# Try Wayland first, fallback to Qt5Agg
try:
    os.environ['QT_QPA_PLATFORM'] = 'wayland;xcb'
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')  # Fallback to TkAgg if Qt fails
import matplotlib.pyplot as plt

# Rest of your plotting code remains the same


#Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

#show first few lines of encoder
print(y[:10])

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Define the model layers
input_layer = tf.keras.layers.Dense(units = 4, activation = 'relu')
layer2 = tf.keras.layers.Dense(units = 8, activation = 'relu')
layer3 = tf.keras.layers.Dense(units = 6, activation = 'relu')
layer4 = tf.keras.layers.Dense(units = 4, activation = 'relu')
output_layer = tf.keras.layers.Dense(units = 3, activation = 'softmax')

# Define the model
model = tf.keras.Sequential([input_layer, layer2, layer3, layer4, output_layer])

# Compile the model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis = 1)

#Side by side comparison
class_names = iris.target_names
y_pred_classes = np.argmax(predictions, axis = 1)
y_test_classes = np.argmax(y_test, axis = 1)
print("\nSide by side comparison:")
for pred, actual in zip(y_pred_classes, y_test_classes):
    print(f"Predicted: {class_names[pred]:20}, Actual: {class_names[actual]}")


# Visualize the training history

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = "validation loss")
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()

#