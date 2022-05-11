import matplotlib as mpl
import numpy as np

from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer
from descent.methods.descent_result import DescentResult
from descent.methods.descent_method import DescentMethod

mpl.use('TkAgg')


def Jacobian(m_f, b, x):
    eps = 1e-6

    if isinstance(x[0], np.float64):
        x = np.array([[p] for p in x])

    grads = []
    for i in range(len(b)):
        t = np.zeros(len(b)).astype(float)
        t[i] = t[i] + eps

        f1 = np.array([m_f(b + t, *p) for p in x])
        f2 = np.array([m_f(b - t, *p) for p in x])

        grad = (f1 - f2) / (2 * eps)
        grads.append(grad)

    return np.column_stack(grads)


def Gauss_Newton(m_f, x, y, b0, epoch):
    tolerance = 1e-5

    if isinstance(x[0], np.float64):
        x = np.array([[p] for p in x])

    points = []

    new = np.array(b0)
    points.append(new.tolist())
    for itr in range(epoch):
        old = new
        J = Jacobian(m_f, old, x)

        dy = y - np.array([m_f(old, *p) for p in x])
        new = old + np.linalg.inv(J.T @ J) @ J.T @ dy

        points.append(new)

        if np.linalg.norm(old - new) < tolerance:
            break

    # return new
    return DescentResult(points, points, m_f, 'Gauss_Newton')


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
    drawer.draw_2d_nonlinear_regression(X, Y, show_image=True)


def two_dim_dataset(result):
    def f(m_b, m_x):
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    data = DatasetReader('planar')
    X, Y = np.array(data.input)[:, 0], np.array(data.output)

    result = Gauss_Newton(f, X, Y, result, epoch=10)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(X, Y, show_image=True)


def three_dim(c, start, stop, size):
    def f(m_b, m_x1, m_x2):
        return m_b[0] - (1 / m_b[1]) * m_x1 ** 2 - (1 / m_b[2]) * m_x2 ** 2

    # Generating data
    X1 = np.linspace(start, stop, size)
    X2 = np.linspace(start, stop, size)
    X1, X2 = np.meshgrid(X1, X2)
    Y = f([2, 3, 1], X1, X2) + np.random.normal(0, 1, size=X1.shape)

    c = Gauss_Newton(f, np.column_stack([X1.ravel(), X2.ravel()]), Y.ravel(), c, 10)
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
    def __init__(self, config):
        config.fistingate()

    def converge(self):
        return DescentResult('pigis')
