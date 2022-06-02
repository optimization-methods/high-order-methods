import numpy as np

from utils import config


class Calculus:
    def __init__(self, xs, ys, r):
        self.__xs = xs
        self.__ys = ys
        self.__r = r

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def dy(self, b):
        return self.__ys - self.__r(b, self.__xs)

    def jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b), dtype=config.dtype)
            t[i] = t[i] + eps
            grad = (self.__r(b + t, self.__xs) - self.__r(b - t, self.__xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def gradient(self, x):
        dy = self.dy(x)
        jacobian = self.jacobian(x)
        return -2 * jacobian.T @ dy
