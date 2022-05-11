import numpy as np


class FuncUtils(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def f(self, point):
        accumulator = 0
        for j in range(0, len(self.y)):
            prediction = point[0]
            for i in range(1, len(point)):
                prediction += self.x[j][i - 1] * point[i]
            accumulator += (self.y[j] - prediction) ** 2
        return accumulator

    def gradient(self, point):
        h = 1e-5
        result = np.zeros(point.size)
        for i, n in enumerate(point):
            point[i] = point[i] + h
            f_plus = self.f(point)
            point[i] = point[i] - 2 * h
            f_minus = self.f(point)
            point[i] = point[i] + h
            result[i] = (f_plus - f_minus) / (2 * h)
        return result
