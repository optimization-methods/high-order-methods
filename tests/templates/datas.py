import numpy as np

from tests.templates.data import Data, ThreeDimensionalData
from utils import config
from utils.dataset_reader import DatasetReader


def polynomial_data(coefficients_number) -> Data:
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
    return Data(f, xs, ys, start)

def fractional_data() -> Data:
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    xs = np.linspace(2, 6, 50)
    ys = f([2, 3], xs) + np.random.normal(0, 0.1, size=len(xs))
    start = [5, 5]
    return Data(f, xs, ys, start)


def polynomial_3d_data() -> ThreeDimensionalData:
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    xs1 = np.linspace(-5, 5, 100)
    xs2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(xs1, xs2)
    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]
    return ThreeDimensionalData(f, xs, ys, start, xs1, xs2)
