from descent.methods.bfgs import BFGSDescentMethod
from descent.methods.descent_method import DescentMethod
from descent.methods.dogleg import DoglegDescentMethod
from descent.methods.gauss_newton import GaussNewtonDescentMethod
from descent.methods.l_bfgs import LBFGSDescentMethod
from tests.templates.data import Data


def dogleg(task: Data) -> DescentMethod:
    return DoglegDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


def gauss(task: Data) -> DescentMethod:
    return GaussNewtonDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


# noinspection SpellCheckingInspection
def bfgs(task: Data) -> DescentMethod:
    return BFGSDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


# noinspection SpellCheckingInspection
def l_bfgs(task: Data) -> DescentMethod:
    return LBFGSDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


def each(task: Data) -> list[DescentMethod]:
    return [
        # dogleg(task),
        # gauss(task),
        # bfgs(task),
        l_bfgs(task),
    ]
