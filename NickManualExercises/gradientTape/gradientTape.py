import tensorflow as tf
import numpy as np

x = tf.constant(3.0)
y = tf.constant(8.0)

#Simple linear regression structure for linear scalar-valued function
def linear(p, q):
    return 2*p + 3*q

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    tape.watch(y)

    # define intermediate variables u and v as functions of base input constants x, y
    u = 3 * x
    v = x + y

    result = linear(u, v)

# compute the gradients
dfdx = tape.gradient(result, x)
dfdy = tape.gradient(result, y)
print("Simple linear regression")
print(f"Function value = {linear(u,v)}, should be 51. {linear(u,v)==51}") # f = 2p + 3q = 2*(3*x) + 3*(x+y) = 2*3*3 + 3*(3+8) = 18+33=51
print(f"dlinear/dx = {dfdx}, should be 9. {linear(u,v)==51}") # df/dx = d/dx[(2p + 3q)] = d/dx[2*(3*x) + 3*(x+y)]=6+3=9
print(f"dlinear/dy = {dfdy}, should be 3. {linear(u,v)==51}") # df/dy = d/dy[(2p + 3q)] = d/dy[2*(3*x) + 3*(x+y)]=3+3=3


# For complex non-linear function
def nonLinearScalarFunc(p, q):
    return p**2 + q**3 + p/q

with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(x)
    tape2.watch(y)

    u = 3 * x
    p = x + y

    result = nonLinearScalarFunc(u,p)

# algebaic simplification
# p**2 + q**3 + p/q = (3*x)**2 + (x+y)***3 + (3x)/(x+y)
# =(3*3)^2 + (3+8)^3 + (3*3)/(3+8)
# = 81 + 1331 + 9/11 =

# compute the gradients
dfdx = tape2.gradient(result, x)
dfdy = tape2.gradient(result, y)
print("Complicated non-linear regression")
print(f"Function value = {nonLinearScalarFunc(u,v)}, should be 1412.82.")
print(f"dnonlinear/dx = {dfdx}, should be approx 420.")
print(f"dnonlinear/dy = {dfdy}, should be approx 366.")

##### For a vector valued function
def vectorValFunc(p: tf.Tensor):
    # x is an input vector
    return tf.stack([p[0]**2, p[1]**3])

z = tf.constant([2.0, 3.0, 6])

with tf.GradientTape(persistent=True) as tape3:
    tape3.watch(z)

    p = z**2
    resultvector = vectorValFunc(p)

# compute the gradients
jacobian = tape3.jacobian(resultvector, z)
print("Vector valued function")
print(f"Function value = {vectorValFunc(p).numpy()}, should be [16,729]") #[(2**2)**2, (3**2)**3] = [16,729]
print(f"Jacobian of vector val function = {jacobian.numpy()}, should be[[32, 0,0 ], [0, 1458, 0].")


x = tf.constant(5.0)
### Compute second partial derivatives
with tf.GradientTape() as outer_tape:
    outer_tape.watch(x)
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(x)
        y = x ** 3
    dy_dx = inner_tape.gradient(y,x)
d2y_dx2 = outer_tape.gradient(dy_dx, x)

# first deriv = 3 x**2; second deriv = 6 * x
# since x = 5, first deriv = 3*25 = 75, second deriv = 6*5 = 30
print(f"first deriv = {dy_dx}, should be 75")
print(f"second deriv = {d2y_dx2}, should be 30")

