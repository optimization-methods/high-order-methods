import numpy as np

from descent.methods.descent_result import DescentResult
from utils import config


class DescentMethod:
    def __init__(self, xs, ys, r):
        self.__xs = np.array(xs, dtype=config.dtype)
        self.__ys = np.array(ys, dtype=config.dtype)
        self.__r = r

    def converge(self) -> DescentResult:
        raise NotImplementedError("You need to inherit this class")

    @property
    def name(self) -> str:
        raise NotImplementedError("You need to inherit this class")

    @property
    def _xs(self):
        return self.__xs

    @property
    def _ys(self):
        return self.__ys

    @property
    def _r(self):
        return self.__r
