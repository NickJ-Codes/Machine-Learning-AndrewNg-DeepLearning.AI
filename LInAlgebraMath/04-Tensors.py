x = 25
print(type(x))

y = 28.0
print(type(y))


# Review scalars in pytorch
import torch
x_pt = torch.tensor(25, dtype = torch.float16)
x_pt
print(x_pt.shape)

# Review scalars in tensorflow
import tensorflow as tf
x_tf = tf.Variable(25, dtype=tf.int16)
x_tf
print(x_tf.shape)
y_tf = tf.Variable(3, dtype=tf.int16)
print(x_tf + y_tf)
print(tf.add(x_tf, y_tf)) # '+' sign and tf.add() are equivalent
