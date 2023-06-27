import numpy as np
import matplotlib.pyplot as plt
from cmath import *


def complex_range(from_re: int, to_re: int, from_im: int, to_im: int, step=0.1) -> np.ndarray:
    re_values = np.arange(from_re, to_re, step)
    im_values = np.arange(from_im, to_im, step)
    arr = np.array([complex(re, im) for re in re_values for im in im_values])
    return arr

class Graph(object):
    def __init__(self, function, re_min=-10, re_max=10, im_min=-10, im_max=10, resolution=8):

        vec_function = np.vectorize(function)
        nums = complex_range(re_min, re_max, im_min, im_max, 1/resolution)

        # y = f(x + iz)

        self.x = np.real(nums)
        self.z = np.imag(nums)

        self.y = vec_function(nums)


    def plot(self, y_min=-10, y_max=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel('Re(x)')
        ax.set_ylabel('y')
        ax.set_zlabel('Im(x)')

        indices = np.where(abs(self.y.imag) < 0.1)
        print(self.y[indices])
        ax.scatter(self.x.real[indices],self.y[indices], self.z[indices], c=self.z[indices], cmap="viridis")

        ax.set_ylim(y_min, y_max)

    def show(self):
        plt.show()

def x_squared():
    graph = Graph(lambda x: x * x, re_min=-15, re_max=15, im_min=-15, im_max=15, resolution=10)
    graph.plot()
    graph.show()


def sin_plot():
    graph = Graph(sin, re_min=-15, re_max=15, im_min=-15, im_max=15, resolution=20)
    graph.plot()
    graph.show()

if __name__ == "__main__":
    sin_plot()


