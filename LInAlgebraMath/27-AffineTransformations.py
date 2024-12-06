import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

vectorV = np.array([3,1])
identityTwo = np.identity(2)
MatrixFlipX = np.array([[1, 0], [0, -1]]) # matrix to flip vector v over x-axis
MatrixStretchX = np.array([[2, 0], [0, 1]]) # matrix to stretch vector v by 2 in x direction
MatrixStretchArbitrary = np.array([[2, 1], [1, 0]]) # matrix to stretch vector v by human specified amount in hardcode
vectorv_rotX = np.matmul(MatrixFlipX, vectorV)
vectorv_stretchX = np.matmul(MatrixStretchX, vectorV)
vectorv_stretchArbitrary = np.matmul(MatrixStretchArbitrary, vectorV)



# Plot vector V and v rot
fig, ax = plt.subplots()
ax.set(xlim=(-10, 10), ylim=(-10, 10),
       xlabel='x', ylabel='y',
       title='Vector V and its rotations') # set all plot properties at once

#Add grid
ax.grid(True)

# Draw x and y axes with improved visibility
ax.axhline(y=0, color='k', linewidth=0.5, zorder=1)
ax.axvline(x=0, color='k', linewidth=0.5, zorder=1)

# Plot both vectors more efficiently
vectors = np.array([[vectorV[0], vectorV[1]],
                    [vectorv_rotX[0], vectorv_rotX[1]],
                   [vectorv_stretchX[0], vectorv_stretchX[1]],
                    [vectorv_stretchArbitrary[0], vectorv_stretchArbitrary[1]]])
colors = ['b', 'r', 'g', 'r']
labels = ['v', 'v_rotX', 'v_stretchX', 'Vector_StretchArbitrary']
for vec, color, label in zip(vectors, colors, labels):
    ax.plot([0, vec[0]], [0, vec[1]],
            color=color,
            label=label,
            linewidth=2,
            zorder=2)

#add legend and lpot
ax.legend()
plt.show()

#Create and concaenate several matrix vectors in numpy
vectorA = np.array([1, 2])
vectorB = np.array([3, 1.5])
vectorC = np.array([1, 2.5])
vectorD = np.array([1.5, 2.5])

MatrixABCD = np.concatenate((np.matrix(vectorA).T,
                            np.matrix(vectorB).T,
                            np.matrix(vectorC).T,
                            np.matrix(vectorD).T),
                            axis=1)
print(f'Matrix ABCD: {MatrixABCD}')

MatrixABCD_Transformed = np.matmul(MatrixStretchArbitrary, MatrixABCD)

fig, ax = plt.subplots()
ax.set(xlim=(-10, 10), ylim=(-10, 10),
       xlabel='x', ylabel='y',
       title='Vector ABCD and its rotations') # set all plot properties at once
ax.grid(True)
ax.axhline(y=0, color='k', linewidth=0.5, zorder=1)
ax.axvline(x=0, color='k', linewidth=0.5, zorder=1)
