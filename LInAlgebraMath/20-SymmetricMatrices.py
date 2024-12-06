# Create identity matrix in tensorflow of dimension 5x5
import tensorflow as tf
tf_identity = tf.eye(5)

#do quick manual tensorflow math
matrixA = tf.constant([[0,1,2], [3,4,5], [6,7,8]])
matrixB = tf.constant([-1,1,-2])
matrixC = tf.constant([[-1,0], [1,1], [-2,2]])


# reshape matrixB into vector
matrixB_reshaped = tf.reshape(matrixB, (3,1))
matrixMultD = tf.matmul(matrixA, matrixB_reshaped)
matrixMultE = tf.matmul(matrixA, matrixC)

print("TF matrix multiplication practice")
print(f'Matrix A * vector B: {matrixMultD}')
print(f'Matrix A * Matrix C: {matrixMultE}')