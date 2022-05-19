import matplotlib as mpl
import numpy as np
import numpy.linalg as ln
from matplotlib import pyplot as plt

from descent.methods.descent_result import DescentResult
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer

mpl.use('TkAgg')


class LineSearch(object):
    def __init__(self, point, direction, X, Y):
        self.beta1 = 0.1  # less -> more possibilities for alpha
        self.beta2 = 0.9  # less -> closer to the minimum
        self.x0 = point
        self.d = direction
        self.X = X
        self.Y = Y

    def line_search1(self, slr):
        print(self.x0)
        print(self.d)

        A = np.linspace(0, 1, 1000)
        B = []
        for i in range(len(A)):
            if self.first(A[i]) and self.second(A[i]):
                B.append(A[i])
        print(len(B))

        if len(B) == 0:
            return slr

        print(B[round(len(B) / 2)])
        return B[round(len(B) / 2)]

    def line_search(self, slr):
        def search(ll, rr):
            mid = (ll + rr) / 2
            while not (self.first(mid) and self.second(mid)):
                if self.first(mid) and not self.second(mid):
                    ll = mid
                else:
                    rr = mid
                mid = (ll + rr) / 2
            return mid

        def find_bound(cond1, cond2, start, sign):
            bound = start
            factor = sign
            while cond1(bound):
                bound += factor
                factor *= 2

            if not cond2(bound):
                self.change_beta()

            return search(start, bound)

        is_first = self.first(slr)
        is_second = self.second(slr)
        if is_first and is_second:
            return slr
        elif is_first and not is_second:
            return find_bound(self.first, self.second, slr, 1)
        elif not is_first and is_second:
            return find_bound(self.second, self.first, slr, -1)
        else:
            self.change_beta()

    def first(self, alpha):
        F_x1 = self.F(self.x0 + alpha * self.d)
        F_x0 = self.F(self.x0)
        return F_x1 <= F_x0 - alpha * self.beta1 * self.gamma(self.x0)

    def second(self, alpha):
        x1 = self.x0 + alpha * self.d
        return self.gamma(x1) <= self.beta2 * self.gamma(self.x0)

    def gamma(self, x):
        grad = -2 * Jacobian(x, self.X).T @ self.r(x)
        return -np.dot(self.d, grad)

    def r(self, point):
        return self.Y - f(point, self.X)

    def F(self, point):
        return np.sum(self.r(point) ** 2)

    @staticmethod
    def change_beta():
        pass
        # print('fuck')


# def f(m_b, m_x):
#     accumulator = 0
#     for i in range(len(m_b)):
#         accumulator += m_b[i] * m_x ** i
#     return accumulator


def f(m_b, m_x):
    return m_b[0] * m_x / (m_b[1] + m_x)


def Jacobian(b, x):
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros(len(b)).astype(float)
        t[i] = t[i] + eps
        grad = (f(b + t, x) - f(b - t, x)) / (2 * eps)
        grads.append(grad)
    return np.column_stack(grads)


def bfgs_method(x0, X, Y, eps=10e-3):
    dy = Y - f(x0, X)
    J = Jacobian(x0, X)
    g = -2 * J.T @ dy

    H = np.eye(len(x0), dtype=int)

    points = [x0]
    while ln.norm(g) > eps:
        direction = -np.dot(H, g)

        alpha = LineSearch(x0, direction, X, Y).line_search1(0.01)

        x1 = x0 + alpha * direction
        step = x1 - x0
        x0 = x1

        dy = Y - f(x1, X)
        J = Jacobian(x1, X)
        new_g = - 2 * J.T @ dy

        g_diff = new_g - g
        g = new_g

        ro = 1.0 / (np.dot(g_diff, step))

        A1 = np.eye(len(x0), dtype=int) - ro * step[:, np.newaxis] * g_diff[np.newaxis, :]
        A2 = np.eye(len(x0), dtype=int) - ro * g_diff[:, np.newaxis] * step[np.newaxis, :]
        H = np.dot(A1, np.dot(H, A2)) + (ro * step[:, np.newaxis] * step[np.newaxis, :])

        points.append(x0.tolist())

    print(points[-1])
    return DescentResult(points, points, f, 'BFGS')


def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # data = DatasetReader('planar').parse()
    # X, Y = np.array(data.input)[:, 0], np.array(data.output)
    # result = bfgs_method(np.ones(5), X, Y)

    X = np.linspace(1, 5, 50)
    Y = f([2, 3], X) + np.random.normal(0, 0.1, size=50)
    result = bfgs_method([10, 10], X, Y)

    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(X, Y, show_image=True)


if __name__ == '__main__':
    main()

# noinspection SpellCheckingInspection
# class BfgsDescentMethod(DescentMethod):
#     def __init__(self, config):
#         config.fistingate()
#
#     def converge(self):
#         return DescentResult('pigis')
