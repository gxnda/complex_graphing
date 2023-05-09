import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from abc import ABC, ABCMeta


def complex_linspace(from_re: int, to_re: int, from_im: int, to_im: int, step=0.1) -> np.array:
    print("Generating complex values...")
    arr = np.array([])
    for re in np.arange(from_re, to_re, step):
        for im in np.arange(from_im, to_im, step):
            arr = np.append(arr, complex(re, im))
    print("Done!")
    return arr


class LightweightGraph(ABC, metaclass=ABCMeta):
    def __init__(self, function, x_range_from: float = 2, x_range_to: float = -2):
        super().__init__()
        self.vec_function = np.vectorize(function)
        self.x = np.linspace(x_range_from, x_range_to, 10000)
        self.y = np.real(self.vec_function(self.x))
        self.z = np.imag(self.vec_function(self.x))
        print(self.y, self.z)

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


class Graph:
    def __init__(self, function, re_min, re_max, im_min, im_max, resolution):
        self.vec_function = np.vectorize(function)
        linspace = complex_linspace(re_min, re_max, im_min, im_max, 1/resolution)
        self.x = self.vec_function(linspace)
        self.y = np.real(linspace)
        self.z = np.imag(linspace)

    def plot(self, y_lim_neg=-10, y_lim_pos=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Re(x)')
        ax.set_ylabel('y')
        ax.set_zlabel('Im(x)')
        ax.scatter(self.y, self.x.real, self.z, c=self.x.imag, cmap='viridis')
        ax.set_ylim(y_lim_neg, y_lim_pos)


    def show(self):
        print(f"Re(x): {self.y}\n\ny: {self.x}\n\nIm(x): {self.z}")
        plt.show()



if __name__ == "__main__":
    graph = Graph(function=tan, re_min=-10, re_max=10, im_min=-1, im_max=1, resolution=30)
    graph.plot()
    graph.show()
