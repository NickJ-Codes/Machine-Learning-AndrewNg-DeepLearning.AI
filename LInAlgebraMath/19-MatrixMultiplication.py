import numpy
import numpy as np

matrixA= numpy.array([[1,2,3], [3,4,5]]) #Dimensions = 2x3
matrixB = numpy.array([1,2,3]) # Dim = 1x3
matrixC_dot = numpy.dot(matrixA, matrixB) #matrix multiplication using dot
matrixC_Matmult = numpy.matmul(matrixA, matrixB) #matrix multiplication using matmul

print("Numpy")
# print("MatrixA: ", matrixA)
# print("matrixB: ", matrixB)
print(f"Matrix multiplied using numpy dot: {matrixC_dot}")
print(f"Matrix multiplied using numpy matmult: {matrixC_Matmult}")

#doing matrix multiplication using tensorflow
import tensorflow as tf
tensorA = tf.constant([[1, 2, 3],
                       [3,4,5]], dtype = float)
tensorB = tf.constant([1, 2, 3], dtype = float)
Tensor_MatrixVectorMult = tf.linalg.matvec(tensorA, tensorB)
tensorB_reshaped = tf.reshape(tensorB, (3,1))
Tensor_MatrixMatrixMult = tf.linalg.matmul(tensorA, tensorB_reshaped)
Tensor_MatrixMatrixMult_SimpleNotation = tensorA @ tensorB_reshaped

print("Tensorflow")
# print("TensorA: ", tensorA)
# print("TensorB: ", tensorB)
print(f"Tensor multiplied using tensorflow matvec: {Tensor_MatrixVectorMult}")
print(f"Tensor multiplied using tensorflow matmul: {Tensor_MatrixMatrixMult}")
print(f"Tensor multiplied using tensorflow matmul using @ notation: {Tensor_MatrixMatrixMult_SimpleNotation}")

#Convert numpy matrix to torch
import torch
torch_matrix = torch.from_numpy(matrixA)
print(f"torch matrix from numpy: {torch_matrix}")