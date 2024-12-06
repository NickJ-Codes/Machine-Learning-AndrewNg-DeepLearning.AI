#Calculate frobenius norm (aka euclidean norm)

import numpy as np
import tensorflow as tf

#using numpy
matrixA= np.array([[1, 2, 8], [3,4,5]])
manual_calc = np.sqrt(np.sum(1*1+2*2+8*8+3*3+4*4+5*5))
#use numpy to calculate froebnius norm
frobenius_norm = np.linalg.norm(matrixA)

print("Frobenius norm math numpy")
print(f"Original Matrix: {matrixA}")
print(f"Manual calculation of Frobenius norm: {manual_calc}")

#using tensorflow
tf_matrixA = tf.constant([[1, 2, 8], [3, 4, 5]], dtype = tf.float16)
# tf_frobenius_norm = tf.norm(tf_matrixA, ord='euclidean')
tf_frobenius_norm = tf.norm(tf_matrixA) # by default, ord = eucdlidean so above is not needed

print("Frobenius norm tensorflow")
print(f"Original Matrix: {tf_matrixA}")
print(f"Tensorflow calculation of Frobenius norm: {tf_frobenius_norm}")



