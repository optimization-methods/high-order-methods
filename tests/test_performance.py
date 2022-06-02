import tracemalloc
from datetime import datetime

from matplotlib import pyplot as plt

from tests.templates import datas, methods
from tests.templates.data import Data


def test_time(data: Data, epoch=100):
    def measure_time(method):
        start_time = datetime.now()
        for i in range(epoch):
            method.converge()
        end_time = datetime.now()
        return (end_time - start_time).total_seconds()

    measurements = [(method.name, measure_time(method)) for method in methods.each(data)]

    plt.title(f'Time (s), {epoch=}')
    plt.bar([item[0] for item in measurements], [item[1] for item in measurements])
    plt.show()


def test_memory(data: Data, epoch=100):
    def measure_memory(method):
        tracemalloc.start()
        for i in range(epoch):
            method.converge()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024

    measurements = [(method.name, measure_memory(method)) for method in methods.each(data)]

    plt.title(f'Memory (KB), {epoch=}')
    plt.bar([item[0] for item in measurements], [item[1] for item in measurements])
    plt.show()


def test_performance():
    data = datas.polynomial_data(5)
    test_time(data, 100)
    test_memory(data, 100)


if __name__ == "__main__":
    test_performance()
