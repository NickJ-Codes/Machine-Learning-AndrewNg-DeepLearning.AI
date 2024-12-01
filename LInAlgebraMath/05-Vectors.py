import tensorflow as tf
import numpy as np
import torch

# Use numpy to create vectors
x = np.array ( [1, 2, 3], dtype=np.int8)
print(x.shape)
print(x)
x_T = x.T #numpy transpose
print(x_T.shape)
print(x_T)
# since x is a 1-d array, then x.T is also a 1-d array with same shape


'''
But if the vector is denoted using double square brackets,
then the shape and transpose are different, then if specified
using the single square bracket notation above
zince numpy has 2 dimensions to play with
'''
y = np.array ([[1, 2, 3]], dtype=np.int8)
print(y.shape)
print(y.T.shape)
print(y)
print(y.T)

# Use pytorch to crate a vector
x_pt = torch.tensor([1, 2, 3], dtype=torch.int8)
print(x_pt.shape)
print(x_pt.T.shape)


# Use tensorflow to create vectors
x_tf = tf.Variable([[1, 2, 3]], dtype=tf.int8)
print(x_tf.shape)
print(tf.transpose(x_tf).shape)
