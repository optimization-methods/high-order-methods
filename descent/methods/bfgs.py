import matplotlib as mpl
import numpy as np
import numpy.linalg as ln

from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer

mpl.use('TkAgg')


# class LineSearch(object):
#     def __init__(self, start, direction, X, Y):
#         self.beta1 = 0.1  # less -> more possibilities for alpha
#         self.beta2 = 0.9  # less -> closer to the minimum
#         self.x0 = start
#         self.d = direction
#         self.X = X
#         self.Y = Y
#
#     def line_search1(self, slr):
#         print(self.x0)
#         print(self.d)
#
#         A = np.linspace(0, 1, 1000)
#         B = []
#         for i in range(len(A)):
#             if self.first(A[i]) and self.second(A[i]):
#                 B.append(A[i])
#         print(len(B))
#
#         if len(B) == 0:
#             return slr
#
#         print(B[round(len(B) / 2)])
#         return B[round(len(B) / 2)]

# def line_search(f, x, p, nabla):
#     a = 1
#     c1 = 1e-4
#     c2 = 0.9
#     fx = f(x)
#     x_new = x + a * p
#     nabla_new = grad(f, x_new)
#     while f(x_new) >= fx + (c1 * a * nabla.T @ p) or nabla_new.T @ p <= c2 * nabla.T @ p:
#         a *= 0.5
#         x_new = x + a * p
#         nabla_new = grad(f, x_new)
#     return a

# def line_search(self, slr):
#     def search(ll, rr):
#         mid = (ll + rr) / 2
#         while not (self.first(mid) and self.second(mid)):
#             if self.first(mid) and not self.second(mid):
#                 ll = mid
#             else:
#                 rr = mid
#             mid = (ll + rr) / 2
#         return mid
#
#     def find_bound(cond1, cond2, start, sign):
#         bound = start
#         factor = sign
#         while cond1(bound):
#             bound += factor
#             factor *= 2
#
#         if not cond2(bound):
#             self.change_beta()
#
#         return search(start, bound)
#
#     is_first = self.first(slr)
#     is_second = self.second(slr)
#     if is_first and is_second:
#         return slr
#     elif is_first and not is_second:
#         return find_bound(self.first, self.second, slr, 1)
#     elif not is_first and is_second:
#         return find_bound(self.second, self.first, slr, -1)
#     else:
#         self.change_beta()
#
# def first(self, alpha):
#     F_x1 = self.F(self.x0 + alpha * self.d)
#     F_x0 = self.F(self.x0)
#     return F_x1 <= F_x0 - alpha * self.beta1 * self.gamma(self.x0)
#
# def second(self, alpha):
#     x1 = self.x0 + alpha * self.d
#     return self.gamma(x1) <= self.beta2 * self.gamma(self.x0)
#
# def gamma(self, x):
#     grad = -2 * Jacobian(x, self.X).T @ self.r(x)
#     return -np.dot(self.d, grad)
#
# def r(self, point):
#     return self.Y - f(point, self.X)
#
# def F(self, point):
#     return np.sum(self.r(point) ** 2)
#
# @staticmethod
# def change_beta():
#     pass


# noinspection PyPep8Naming
class BfgsDescentMethod(object):
    class LineSearch(object):
        # noinspection SpellCheckingInspection
        def __init__(self, point, d, bfgs):
            self.point = point
            self.d = d
            self.bfgs = bfgs

        def g(self, alpha):
            return self.bfgs.f(self.point + alpha * self.d)

        def grad(self, alpha):
            return self.bfgs.gradient(self.point + alpha * self.d)

        def data(self, alpha):
            return self.g(alpha), self.grad(alpha).T @ self.d

        def wolfe(self, epoch=20, alpha=1, c1=1e-4, c2=0.9, rho=0.8):
            k = 0
            g_0, gamma_0 = self.data(0)
            while True:
                g_alpha, gamma_alpha = self.data(alpha)
                first = (g_alpha > g_0 + c1 * (alpha * gamma_0))
                second = (gamma_alpha < c2 * gamma_0)
                if (k < epoch) and (first or second):
                    break
                alpha *= rho
                k += 1
            return alpha

    def __init__(self, r, start, xs, ys):
        self.start = start
        self.r = r
        self.xs = xs
        self.ys = ys
        self.eps = 10e-3

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def dy(self, b):
        return self.ys - self.r(b, self.xs)

    def jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r(b + t, self.xs) - self.r(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def gradient(self, x):
        dy = self.dy(x)
        jacobian = self.jacobian(x)
        return -2 * jacobian.T @ dy

    def converge(self):
        g = self.gradient(self.start)
        I = np.eye(len(self.start), dtype=config.dtype)
        H = I
        points = [self.start]
        x0 = points[-1]
        while ln.norm(g) > self.eps:
            direction = -np.dot(H, g)

            alpha = self.LineSearch(x0, direction, self).wolfe()

            x1 = x0 + alpha * direction
            step = x1 - x0
            x0 = x1

            new_g = self.gradient(x1)
            g_diff = new_g - g
            g = new_g

            ro = 1.0 / (np.dot(g_diff, step))

            A1 = I - ro * step[:, np.newaxis] * g_diff[np.newaxis, :]
            A2 = I - ro * g_diff[:, np.newaxis] * step[np.newaxis, :]
            H = np.dot(A1, np.dot(H, A2)) + (ro * step[:, np.newaxis] * step[np.newaxis, :])

            points.append(x0.tolist())

        return DescentResult(points, points, self.r, 'BFGS')


def main():
    # def r(m_b, m_x):
    #     accumulator = 0
    #     for i in range(len(m_b)):
    #         accumulator += m_b[i] * m_x ** i
    #     return accumulator

    def r(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # data = DatasetReader('planar').parse()
    # xs, ys = np.array(data.input)[:, 0], np.array(data.output)
    # result = BfgsDescentMethod(r, np.ones(10), xs, ys).evaluate()

    xs = np.linspace(1, 5, 50)
    ys = r([2, 3], xs) + np.random.normal(0, 0.1, size=50)
    result = BfgsDescentMethod(r, [10, 10], xs, ys).converge()

    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


if __name__ == '__main__':
    main()
