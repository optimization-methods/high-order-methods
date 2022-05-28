import numpy as np

from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


# mpl.use('TkAgg')
# def Jacobian(m_f, b, x):
#     eps = 1e-6
#     grads = []
#     for i in range(len(b)):
#         t = np.zeros(len(b)).astype(float)
#         t[i] = t[i] + eps
#         grad = (m_f(b + t, x) - m_f(b - t, x)) / (2 * eps)
#         grads.append(grad)
#
#     return np.column_stack(grads)
# def Gauss_Newton(m_f, x, y, b0, epoch):
#     tolerance = 1e-5
#
#     points = []
#
#     new = np.array(b0)
#     points.append(new.tolist())
#     for itr in range(epoch):
#         old = new
#         J = Jacobian(m_f, old, x)
#         dy = y - m_f(old, x)
#         new = old + np.linalg.inv(J.T @ J) @ J.T @ dy
#         points.append(new)
#
#         if np.linalg.norm(old - new) < tolerance:
#             break
#
#     return DescentResult(points, points, m_f, 'Gauss_Newton')

def two_dim(initial, start, stop, size):
    # TODO: can be applied draw_3d
    # predicted height function
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    # Generating data
    X = np.linspace(start, stop, size)
    Y = f([2, 3], X) + np.random.normal(0, 0.1, size=size)

    result = Gauss_Newton(f, X, Y, initial, epoch=10)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(show_image=True)


def two_dim_dataset(result):
    def f(m_b, m_x):
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    data = DatasetReader('planar').parse()
    X, Y = np.array(data.input)[:, 0], np.array(data.output)

    result = Gauss_Newton(f, X, Y, result, epoch=10)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(show_image=True)


def three_dim(c, start, stop, size):
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    # Generating data
    X1 = np.linspace(start, stop, size)
    X2 = np.linspace(start, stop, size)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    Y = f([2, 3, 1], X) + np.random.normal(0, 1, size=len(X))

    c = Gauss_Newton(f, X, Y, c, 10)
    drawer = Drawer(c)
    drawer.draw_3d_nonlinear_regression(X1, X2, Y, show_image=True)


def main():
    np.set_printoptions(suppress=True)
    two_dim_dataset(np.ones(10))
    two_dim([1, 1], 1, 5, 50)
    three_dim([1, 1, 1], -5, 5, 100)


if __name__ == '__main__':
    main()


class GaussNewtonDescentMethod(DescentMethod):
    def __init__(self, func, start, xs, ys,
                 epoch=30,
                 tolerance=1e-5):
        self.r = func
        self.start = start
        self.xs = np.array(xs, config.dtype)
        self.ys = np.array(ys, config.dtype)

        self.epoch = epoch
        self.tolerance = tolerance

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def dy(self, b):
        return self.ys - self.r(b, self.xs)

    def get_jacobian(self, b, x):
        eps = 1e-6
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r(b + t, x) - self.r(b - t, x)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def converge(self):
        points = []
        new = np.array(self.start)
        points.append(new.tolist())
        for itr in range(self.epoch):
            old = new
            jacobian = self.get_jacobian(old, self.xs)
            dy = self.dy(old)
            new = old + np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ dy
            points.append(new.tolist())

            if np.linalg.norm(old - new) < self.tolerance:
                break

        return DescentResult(self.f, points, points, r=self.r, method_name='Gauss_Newton')
