from descent.methods.bfgs import BfgsDescentMethod
from descent.methods.descent_method import DescentMethod
from descent.methods.dogleg import DoglegDescentMethod
from descent.methods.gauss_newton import GaussNewtonDescentMethod
from descent.methods.l_bfgs import LBfgsDescentMethod
from tasks.templates.task import Task


def dogleg(task: Task) -> DescentMethod:
    return DoglegDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


def gauss(task: Task) -> DescentMethod:
    return GaussNewtonDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


# noinspection SpellCheckingInspection
def bfgs(task: Task) -> DescentMethod:
    return BfgsDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


# noinspection SpellCheckingInspection
def l_bfgs(task: Task) -> DescentMethod:
    return LBfgsDescentMethod(task.f, start=task.start, xs=task.xs, ys=task.ys)


def each(task: Task) -> list[DescentMethod]:
    return [
        dogleg(task),
        gauss(task),
        bfgs(task),
        l_bfgs(task),
    ]
