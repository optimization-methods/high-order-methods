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
    result = m_method.converge()
    print(f'{m_method.name} {result.rescaled_scalars[-1]}')
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


def draw_3d_nonlinear_regression(method, xs1, xs2, y):
    print('testing...')
    result = method.converge()
    print(f'{method.name} {result.rescaled_scalars[-1]}')
    drawer = Drawer(result)
    drawer.draw_3d_nonlinear_regression(xs1, xs2, y, show_image=True)


def draw_2d_converge_projection(method):
    print('testing...')
    result = method.converge()
    print(f'{method.name} {result.rescaled_scalars[-1]}')
    drawer = Drawer(result)
    drawer.draw_2d(True)

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

def test_2d_linear_regression():
    data = polynomial_data(2)
    xs, ys = data.xs, data.ys

    for method in methods.each(data):
        draw_2d_nonlinear_regression(method, xs, ys)
        draw_2d_converge_projection(method)

def test_2d_fractional_approximation():
    xs = np.linspace(2, 6, 50)
    data = fractional_data(xs)
    xs, ys = xs, data.ys

    for method in methods.each(data):
        draw_2d_nonlinear_regression(method, xs, ys)

def test_2d_polynomial_approximation():
    for method in methods.each(polynomial_data(5)):
        draw_2d_nonlinear_regression(method, method.xs, method.ys)


def test_3d_polynomial_approximation():
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1, x2)

    data = polynomial_3d_data(x1, x2)
    xs, ys = data.xs, data.ys

    for method in methods.each(data):
        draw_3d_nonlinear_regression(method, x1, x2, ys)


if __name__ == "__main__":
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
