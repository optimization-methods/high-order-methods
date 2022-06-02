import tracemalloc
from datetime import datetime
from typing import Callable

from matplotlib import pyplot as plt

from descent.methods.descent_method import DescentMethod
from tests.templates import datas, methods
from tests.templates.data import Data

def measure_time(method, epoch):
    start_time = datetime.now()
    print(method.name)
    for i in range(epoch):
        method.converge()
    end_time = datetime.now()
    return (end_time - start_time).total_seconds()

def measure_memory(method, epoch):
    tracemalloc.start()
    print(method.name)
    for i in range(epoch):
        method.converge()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024

def test_measure(data: Data, measure: Callable[[DescentMethod, int], float], name: str, epoch: int):
    measurements = [(method.name, measure(method, epoch)) for method in methods.each(data)]

    plt.title(f'{name}, {epoch=}')
    plt.bar([item[0] for item in measurements], [item[1] for item in measurements])
    plt.show()

def test_time(data: Data, epoch=100):
    print(f'===TIME===')
    return test_measure(data, measure_time, 'Time (s)', epoch)

def test_memory(data: Data, epoch=100):
    print(f'===MEMORY===')
    return test_measure(data, measure_memory, 'Memory (KB)', epoch)

def test_performance():
    data = datas.polynomial_data(5)
    test_time(data, 1)
    test_memory(data, 1)


if __name__ == "__main__":
    test_performance()
