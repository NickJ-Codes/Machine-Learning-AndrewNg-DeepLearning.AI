import tensorflow as tf

tf_matrixA = tf.constant([[1, 2], [3, 4]])
tf_matrixB = tf.constant([[5, 6], [7, 8]])

# Calculate Tranpose of matrix A
tf_matrixA_Trans = tf.transpose(tf_matrixA)

print(f"matrix A: {tf_matrixA}")
print(f"Transpose of matrix A: {tf_matrixA_Trans}")

#calculate hadamard product of matrix a and b
tf_hadamardAB = tf.multiply(tf_matrixA, tf_matrixB)
tf_hadamardBA = tf.multiply(tf_matrixB, tf_matrixA)
print(f"Transpose of matrix AxB: {tf_hadamardAB}")
print(f"Transpose of matrix BxA: {tf_hadamardBA}")

vectorA = tf.constant([1, 2, 3])
vectorB = tf.constant([4, 5, 6])
tf.tensordot = tf.tensordot(vectorA, vectorB, axes=1)
print(f"Dot product of vectorA and vectorB: {tf.tensordot}") # should be 32