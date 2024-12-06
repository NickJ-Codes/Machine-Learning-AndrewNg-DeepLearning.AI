import numpy
import numpy as np
import tensorflow as tf

numpyX = np.array([0,1,2])
numpyY = np.array([2,3,4])
numpyZ = np.dot(numpyX, numpyY)
print("Numpy dot product: ", numpyZ)

tensorX = tf.constant([0, 1, 2])
tensorY = tf.constant([2, 3, 4])
tensorZ = tf.tensordot(tensorX, tensorY, axes=1)
print("Tensorflow dot product: ", tensorZ.numpy())
