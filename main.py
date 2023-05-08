import numpy as np
import matplotlib.pyplot as plt
from cmath import sqrt, asin, sin
from abc import ABC, ABCMeta
from math import floor, ceil, pi


def all_asin(num: float, min_real: float=-10, max_real: float=10) -> complex:
    """NOT WORKING: Returns multiple values instead of just 1 because cmath.asin() bad"""

    first_asin = asin(num)
    first_re = first_asin.real
    first_im = first_asin.imag

    smallest_n = floor(min_real/(2*pi))
    largest_n = ceil(max_real/(2*pi))
    all_re = [first_re + 2*pi*n for n in range(smallest_n, largest_n)]

    all_results = [complex(re, first_im) for re in all_re]
    return all_results[0]


class Graph(ABC, metaclass=ABCMeta):
    def __init__(self, function, x_range_from: float=2, x_range_to: float=-2):
        super().__init__()
        self.vec_function = np.vectorize(function)
        self.x = np.linspace(x_range_from, x_range_to, 10000)
        self.y = np.real(self.vec_function(self.x))
        self.z = np.imag(self.vec_function(self.x))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Re(x)')
        ax.set_ylabel('y')
        ax.set_zlabel('Im(x)')
        ax.plot(self.y, self.x, self.z, color="blue")

    @staticmethod
    def show():
        plt.show()


if __name__ == "__main__":
    sin_graph = Graph(function=asin)
    sin_graph.plot()
    sin_graph.show()
