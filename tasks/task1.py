import tracemalloc
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from tasks.templates import methods
from tasks.templates.task import Task
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


# mpl.use("TkAgg")


def draw_2d_nonlinear_regression(m_method, xs, ys):
    print('testing...')
    result = method.converge()
    print(f'{result.method_name} {result.rescaled_scalars[-1]}')
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


def draw_3d(method, xs1, xs2, y):
    drawer = Drawer(method.converge())
    drawer.draw_3d_nonlinear_regression(xs1, xs2, y, show_image=True)


def draw(method):
    drawer = Drawer(method.converge())
    drawer.draw_2d(True)


def test_complexity(method, epoch):
    tracemalloc.start()
    start_time = datetime.now()

    for i in range(epoch):
        method.converge()

    end_time = datetime.now()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end_time - start_time).total_seconds() * 1000 / epoch, peak / 1024


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


# def test4_linear_regression_perfomance():
#     data = polynomial_data(2)
#     # noinspection SpellCheckingInspection
#     measurements = [
#         [test_complexity(data, dogleg, 1000), 'Dogleg'],
#         [test_complexity(data, gauss, 1000), 'Gauss'],
#     ]
#     # noinspection SpellCheckingInspection
#     plt.title('Time (ms)')
#     names = [item[1] for item in measurements]
#     times = [item[0][0] for item in measurements]
#     memories = [item[0][1] for item in measurements]
#     plt.bar(names, times)
#     plt.show()
#     plt.title('Memory (KB)')
#     plt.bar(names, memories)
#     plt.show()

def polynomial_3d_data(x1, x2):
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]
    return Task(f, xs, ys, start)

def test_time(data: Task):
    def measure_time(method, epoch=1000):
        start_time = datetime.now()
        for i in range(epoch):
            method.converge()
        end_time = datetime.now()
        return (end_time - start_time).total_seconds()

    measurements = [(method.name, measure_time(method)) for method in methods.each(data)]

    plt.title('Time (s)')
    plt.bar([item[0] for item in measurements], [item[1] for item in measurements])
    plt.show()

def test_memory(data: Task):
    def measure_memory(method, epoch=1000):
        tracemalloc.start()
        for i in range(epoch):
            method.converge()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024

    measurements = [(method.name, measure_memory(method)) for method in methods.each(data)]

    plt.title('Memory (KB)')
    plt.bar([item[0] for item in measurements], [item[1] for item in measurements])
    plt.show()


def fractional_data(xs):
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    ys = f([2, 3], xs) + np.random.normal(0, 0.1, size=len(xs))
    start = [5, 5]
    return Task(f, xs, ys, start)


def polynomial_3d_data(x1, x2):
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]
    return Task(f, xs, ys, start)


def test1():
    coefficients_number = 2
    data = polynomial_data(coefficients_number)
    xs, ys = data.xs, data.ys

    # draw_2d(dogleg(data), xs, ys)
    # draw_2d(gauss(data), xs, ys)
    # draw_2d(bfgs(data), xs, ys)
    draw_2d(l_bfgs(data), xs, ys)
    draw(l_bfgs(data))


def test2():
    xs = np.linspace(2, 6, 50)
    data = fractional_data(xs)
    xs, ys = xs, data.ys

    # draw_2d(bfgs(data), xs, ys)
    # draw_2d(dogleg(data), xs, ys)
    # draw_2d(gauss(data), xs, ys)
    draw_2d(l_bfgs(data), xs, ys)


def test3():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1, x2)

    data = polynomial_3d_data(x1, x2)
    xs, ys = data.xs, data.ys

    for method in methods.each(data):
        draw_3d_nonlinear_regression(method, x1, x2, ys)


if __name__ == "__main__":
    # fixme урод давай без цифр, просто пиши словами
    test_linear_regression()
    test_non_linear_regression()
    test_non_linear_regression_perfomance()
    # test1()
    # test_compare()
    # test2()
    # test3()
    # test_stat()
    # test4_linear_regression_perfomance()
    # test5()
    # test_4_polynomial()
