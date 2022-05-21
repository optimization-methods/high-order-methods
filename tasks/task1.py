import numpy as np

from descent.methods.bfgs import BfgsDescentMethod
from descent.methods.dogleg import DoglegDescentMethod
from descent.methods.gauss_newton import GaussNewtonDescentMethod
from descent.methods.l_bfgs import LBfgsDescentMethod
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


def dogleg(xs, ys, f, start):
    return DoglegDescentMethod(f, start=start, xs=xs, ys=ys)


def gauss(xs, ys, f, start):
    return GaussNewtonDescentMethod(f, start=start, xs=xs, ys=ys)


# noinspection SpellCheckingInspection
def bfgs(xs, ys, f, start):
    return BfgsDescentMethod(f, start=start, xs=xs, ys=ys)


def l_bfgs(xs, ys, f, start):
    return LBfgsDescentMethod(f, start=start, xs=xs, ys=ys)


def draw_2d(method, xs, ys):
    print('testing...')
    result = method.converge()
    print(result.method_name + ' converged')
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


def draw_3d(method, xs1, xs2, y):
    drawer = Drawer(method.converge())
    drawer.draw_3d_nonlinear_regression(xs1, xs2, y, show_image=True)


def test1():
    def f(m_b, m_x):
        m_x = np.array(m_x, dtype=config.dtype)
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    data = DatasetReader('planar').parse()
    xs = np.array(data.input)[:, 0]
    ys = data.output
    start = np.ones(10)

    draw_2d(dogleg(xs, ys, f, start), xs, ys)
    draw_2d(gauss(xs, ys, f, start), xs, ys)


def test2():
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    size = 50
    xs = np.linspace(1, 5, size)
    ys = f([2, 3], xs) + np.random.normal(0, 0.1, size=size)
    start = [1, 1]

    draw_2d(dogleg(xs, ys, f, start), xs, ys)
    draw_2d(gauss(xs, ys, f, start), xs, ys)
    draw_2d(bfgs(xs, ys, f, start), xs, ys)


def test3():
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1, x2)
    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]

    draw_3d(dogleg(xs, ys, f, start), x1, x2, ys)
    draw_3d(gauss(xs, ys, f, start), x1, x2, ys)
    draw_3d(bfgs(xs, ys, f, start), x1, x2, ys)


if __name__ == "__main__":
    test1()
    test2()
    test3()
