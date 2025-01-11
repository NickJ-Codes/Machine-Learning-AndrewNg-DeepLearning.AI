import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 10, activation = 'softmax')
])
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(...,BinaryCrossentropy)
