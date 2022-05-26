import tracemalloc
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from descent.methods.bfgs import BfgsDescentMethod
from descent.methods.dogleg import DoglegDescentMethod
from descent.methods.gauss_newton import GaussNewtonDescentMethod
from descent.methods.l_bfgs import LBfgsDescentMethod
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


def dogleg(task):
    return DoglegDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)

def gauss(task):
    return GaussNewtonDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)

def bfgs(task):
    return BfgsDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)

def l_bfgs(task):
    return LBfgsDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


def draw_2d(method, xs, ys):
    print('testing...')
    result = method.converge()
    print(result.method_name + ' converged')
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


def draw_3d(method, xs1, xs2, y):
    drawer = Drawer(method.converge())
    drawer.draw_3d_nonlinear_regression(xs1, xs2, y, show_image=True)

def test_complexity(data, method, epoch):
    tracemalloc.start()
    start_time = datetime.now()

    for i in range(epoch):
        result = method(data)

    end_time = datetime.now()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end_time - start_time).total_seconds() * 1000 / epoch, peak / 1024

def test4():
    data = fractional_data(np.linspace(1, 5, 50))
    # noinspection SpellCheckingInspection
    measurements = [
        [test_complexity(data, dogleg, 1000), 'Dogleg'],
        [test_complexity(data, gauss, 1000), 'Gauss'],
        [test_complexity(data, bfgs, 1000), 'BFGS'],
        [test_complexity(data, l_bfgs, 1000), 'L-BFGS']
    ]
    # noinspection SpellCheckingInspection
    plt.title('Time (ms)')
    names = [item[1] for item in measurements]
    times = [item[0][0] for item in measurements]
    memories = [item[0][1] for item in measurements]
    plt.bar(names, times)
    plt.show()
    plt.title('Memory (KB)')
    plt.bar(names, memories)
    plt.show()

@dataclass
class Task:
    f: object
    xs: object
    ys: object
    start: object


def polynomial_data(coefficients_number):
    def f(m_b, m_x):
        m_x = np.array(m_x, dtype=config.dtype)
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    data = DatasetReader('planar').parse()
    xs = np.array(data.input)[:, 0]
    ys = data.output
    start = np.ones(coefficients_number)
    return Task(f, xs, ys, start)

def fractional_data(xs):
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    ys = f([2, 3], xs) + np.random.normal(0, 0.1, size=len(xs))
    start = [5, 5]
    return Task(f, xs, ys, start)

def polynomial_3d_data(x1, x2):
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    x1, x2 = np.meshgrid(x1, x2)
    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]
    return Task(f, xs, ys, start)

def test1():
    coefficients_number = 2
    data = polynomial_data(coefficients_number)
    xs, ys = data.xs, data.ys

    draw_2d(dogleg(data), xs, ys)
    draw_2d(gauss(data), xs, ys)
    draw_2d(bfgs(data), xs, ys)
    draw_2d(l_bfgs(data), xs, ys)


def test2():
    xs = np.linspace(1, 5, 50)
    data = fractional_data(xs)
    xs, ys = xs, data.ys

    draw_2d(dogleg(data), xs, ys)
    draw_2d(gauss(data), xs, ys)
    draw_2d(bfgs(data), xs, ys)
    draw_2d(l_bfgs(data), xs, ys)


def test3():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    data = polynomial_3d_data(x1, x2)
    xs, ys = data.xs, data.ys

    draw_3d(dogleg(data), x1, x2, ys)
    draw_3d(gauss(data), x1, x2, ys)
    draw_3d(bfgs(data), x1, x2, ys)
    draw_3d(l_bfgs(data), x1, x2, ys)


if __name__ == "__main__":
    # test1()
    test2()
    # test3()
    # test4()
