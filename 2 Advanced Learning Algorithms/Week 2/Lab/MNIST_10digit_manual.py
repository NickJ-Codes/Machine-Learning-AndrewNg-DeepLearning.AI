# import MNIST 10 digit data set
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from  tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print the shapes to verify the data
print('Training Images Shape:', X_train.shape)    # Should be (60000, 28, 28)
print('Training Labels Shape:', y_train.shape)    # Should be (60000,)
print('Test Images Shape:', X_test.shape)         # Should be (10000, 28, 28)
print('Test Labels Shape:', y_test.shape)         # Should be (10000,)

#normalize values from 0 to 1
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshape 28x28 pixel into 1 d input array
X_train_onedim = X_train.reshape(X_train.shape[0], -1)
X_test_onedim = X_test.reshape(X_test.shape[0], -1)
print(f"Reshaped Training set shape: {X_train_onedim.shape}")

#one hot encode the labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

#Let's define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation= 'relu', input_shape =(784,)),
    tf.keras.layers.Dense(10,activation = 'softmax')
])
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
history = model.fit(
    X_train_onedim,          # Added training data
    y_train_encoded,         # Added training labels
    epochs=1,
    batch_size=256,          # Fixed typo in 'batch_size'
    validation_split=0.2,   # Added validation split
    verbose=1
)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


