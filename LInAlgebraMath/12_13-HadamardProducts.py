import numpy as np
np.random.seed(1000)

import torch
torch.manual_seed(1000)

import tensorflow as tf
tf.random.set_seed(1000)

#Using numpy, Generate 2 random 2x3 matrices with values between -10 and 10 using NumPy. Then calculate the Hademard product
matrix1 = np.random.randint(-10, 11, size=(3, 3))
matrix2 = np.random.randint(-10, 11, size=(3, 3))
# calculate matrix multiplication in numpy of matrix1 and matrix2

hadamard_product = np.multiply(matrix1, matrix2)
numpy_matrix_product = matrix1 @ matrix2 #numpy uses @ sign for matmult
numpy_matrix_productSAME= np.matmul(matrix1, matrix2) #Numpy uses matmal for matrix mult
numpy_matrix_productSAME2= np.dot(matrix1, matrix2) #numpy dot can do both dot product and matrix mult for 2d arrays


print("Numpy outputs")
print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)
print("\nHadamard Product:")
print(hadamard_product)
print("\nMtx Multiplication Product:")
print(numpy_matrix_product)

#Using tensorflow, Generate 2 random 2x3 matrices with values between -10 and 10 using NumPy. Then calculate the Hademard product
matrix1 = tf.random.uniform(shape=(3, 3), minval=-10, maxval=11, dtype=tf.int32)
matrix2 = tf.random.uniform(shape=(3, 3), minval=-10, maxval=11, dtype=tf.int32)
hadamard_product = tf.multiply(matrix1, matrix2)

print("Tensorflow outputs")
print("Matrix 1:")
print(matrix1.numpy())
print("\nMatrix 2:")
print(matrix2.numpy())
print("\nHadamard Product:")
print(hadamard_product.numpy())
#pytorch sum of all elements of matrix1
tf_sum = tf.reduce_sum(matrix1) # sum all items into one scalar
tf_sum = tf.reduce_sum(matrix1, axis = 0) #sum along rows
tf_sum = tf.reduce_sum(matrix1, axis = 1) #sum along columns


#in pytorch, generate 1 random 3x3 matrix
matrix1 = torch.randint(-10, 11, size=(2, 2))
matrix2 = torch.randint(-10, 11, size=(2, 2))
hadamard_product = torch.mul(matrix1, matrix2)
#calculate matrix multiplication of matrix1, matrix2 in pytorch
matrix_product = torch.matmul(matrix1, matrix2)
matrix_sum = torch.sum(matrix1)
print("Pytroch")
print(f"matrix 1: {matrix1}")
print(f"matrix 2: {matrix2}")
print(f"Matrix hadamard product: {hadamard_product}")
print(f"Matrix MatMult product: {matrix_product}")
print(f"Matrix element sum: {matrix_sum}")

