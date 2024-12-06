import numpy as np

#solve simple linear equations using numpy and matrix inversion
# 4b + 2c = 4; -5b - 3c = -7

# This converts into following matrices with unknown coefficient vector w= (b, c)
X = np.array([[4,2], [-5,-3]])
y = np.array([4, -7])

# the linear equation is y = Xw therefore X_inv * y = X_inv * X * w = w
X_inv = np.linalg.inv(X)
w = np.matmul(X_inv, y)

print("Numpy linear equation solver")
print(f"linear equations 4b + 2c = 4; -5b - 3c = -7")
print(f"(b,c): {w}")

# Let's do the same using tensorflow now
import tensorflow as tf
MatrixA = tf.constant([[4, 2], [-5, -3]], dtype = float)
vectorb = tf.constant([4, -7], dtype = float)
Matrix_inv = tf.linalg.inv(MatrixA)
w_tensorflow = tf.linalg.matvec(Matrix_inv, vectorb)

print("Tensorflow linear equation solver")
print(f"linear equations 4b + 2c = 4; -5b - 3c = -7")
print(f"(b, c): {w_tensorflow}")