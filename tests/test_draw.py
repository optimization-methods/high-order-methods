from tests.templates import methods, datas
from utils.drawer import Drawer
import matplotlib as mpl


# mpl.use("TkAgg")

def converge(method):
    print('testing...')
    result = method.converge()
    print(f'{method.name} {result.rescaled_scalars[-1]}')
    drawer = Drawer(result)
    return drawer


def draw_2d_nonlinear_regression(method, xs, ys):
    converge(method).draw_2d_nonlinear_regression(xs, ys, show_image=True)


def draw_3d_nonlinear_regression(method, xs1, xs2, y):
    converge(method).draw_3d_nonlinear_regression(xs1, xs2, y, show_image=True)


def draw_2d_converge_projection(method):
    converge(method).draw_2d(True)


def test_2d_linear_regression():
    data = datas.polynomial_data(2)
    for method in methods.each(data):
        draw_2d_nonlinear_regression(method, data.xs, data.ys)


def test_2d_converge_projection():
    data = datas.polynomial_data(2)
    for method in methods.each(data):
        draw_2d_converge_projection(method)


def test_2d_fractional_approximation():
    data = datas.fractional_data()
    for method in methods.each(data):
        draw_2d_nonlinear_regression(method, data.xs, data.ys)


def test_2d_polynomial_approximation():
    data = datas.polynomial_data(5)
    for method in methods.each(data):
        draw_2d_nonlinear_regression(method, data.xs, data.ys)


def test_3d_polynomial_approximation():
    data = datas.polynomial_3d_data()
    for method in methods.each(data):
        draw_3d_nonlinear_regression(method, data.x1, data.x2, data.ys)


if __name__ == "__main__":
    test_3d_polynomial_approximation()
    test_2d_linear_regression()
    test_2d_polynomial_approximation()
    test_2d_fractional_approximation()
    test_2d_converge_projection()
