import numpy as np
import matplotlib.pyplot as plt
from cmath import sqrt, asin, sin


# Define the function f(x)
def f(x):
    return asin(x)


def neg_f(x):
    return -1 * f(x)


# Create a vectorized version of f(x) for array inputs

f_vec = np.vectorize(f)
x = np.linspace(-2, 2, 10000)
y = np.real(f_vec(x))
z = np.imag(f_vec(x))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y, x, z, color="blue")

# Negative side of the function, ONLY USING THIS FOR + OR - FUNCTIONS (eg x = +-sqrt(y))
"""
f2_vec = np.vectorize(neg_f)
y = np.real(f2_vec(x))
z = np.imag(f2_vec(x))
ax.plot(y, x, z, color="blue")
"""

ax.set_xlabel('Re(x)')
ax.set_ylabel('y')
ax.set_zlabel('Im(x)')
plt.show()
