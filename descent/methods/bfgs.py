import matplotlib as mpl
import numpy as np
import numpy.linalg as ln
import scipy.optimize

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
class BFGS(object):
    def __init__(self, x0, xs, ys):
        self.start = x0
        self.xs = xs
        self.ys = ys
        self.eps = 10e-3

    # def r_func(self, m_b, m_x):
    #     return m_b[0] * m_x / (m_b[1] + m_x)

    def r_func(self, m_b, m_x):
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += 1 / m_b[i] * m_x ** i
        return accumulator

    def f(self, m_b):
        return np.sum(self.dy(m_b) ** 2)

    def dy(self, b):
        return self.ys - self.r_func(b, self.xs)

    def Jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r_func(b + t, self.xs) - self.r_func(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def gradient(self, x):
        dy = self.dy(x)
        J = self.Jacobian(x)
        return -2 * J.T @ dy
    
    def hessian(self, x):
        J = self.Jacobian(x)
        return 2 * J.T @ J

    def wolfe(self, x_k, p_k, alpha=1, c1=1e-4, c2=0.9, rho=0.8):
        k = 0
        maxIt = 20
        jok = self.f(x_k + alpha * p_k)
        cok = self.f(x_k) + c1 * (alpha * self.gradient(x_k).T @ p_k)
        pig = (self.gradient(x_k + alpha * p_k).T @ p_k)
        swi = c2 * (self.gradient(x_k).T @ p_k)
        print('COK', self.gradient(x_k).T, p_k, sep='\n')
        print(jok, cok, pig, swi, sep='\nJOK\n')
        while (k < maxIt) and ((jok > cok) or (pig < swi)):  # condition 2
            alpha *= rho
            k += 1
        return alpha

    def evaluate(self):
        result = [
            np.ones(10).tolist(),
            scipy.optimize.minimize(self.f, self.start, jac=self.Jacobian, method='bfgs').x
        ]

        return DescentResult(result, result, self.r_func, 'PIG')

        # g = self.gradient(self.start)
        # I = np.eye(len(self.start), dtype=config.dtype)
        # H = I
        # points = [self.start]
        # x0 = points[-1]
        # while ln.norm(g) > self.eps:
        #     direction = -np.dot(H, g)
        #
        #     # alpha = LineSearch(x0=x0, xs=self.xs, ys=self.ys, r_func=self.r_func).evaluate()
        #     # alpha = LineSearch(x0, direction, X, Y).line_search1(0.01)
        #     # alpha = scipy.optimize.line_search(self.f, self.gradient, x0, direction)[0]
        #     alpha = self.wolfe(x0, direction)
        #     from scipy.optimize import minimize
        #
        #     x1 = x0 + alpha * direction
        #     step = x1 - x0
        #     x0 = x1
        #
        #     new_g = self.gradient(x1)
        #     g_diff = new_g - g
        #     g = new_g
        #
        #     ro = 1.0 / (np.dot(g_diff, step))
        #
        #     A1 = I - ro * step[:, np.newaxis] * g_diff[np.newaxis, :]
        #     A2 = I - ro * g_diff[:, np.newaxis] * step[np.newaxis, :]
        #     H = np.dot(A1, np.dot(H, A2)) + (ro * step[:, np.newaxis] * step[np.newaxis, :])
        #
        #     points.append(x0.tolist())
        #
        # print(points[-1])
        # return DescentResult(points, points, self.r_func, 'BFGS')

    # def BFGS_пиговый(self):
    #     epsilon = 1e-4  # iteration condtion
    #     m = np.shape(self.start)[0]
    #     # B_k = eye(m)
    #     H_k = np.eye(m)
    #     I = np.eye(m)
    #     y_array = [self.f(self.start)]
    #     k = 1
    #     while abs(self.gradient(self.start)[0]) > epsilon:
    #         g_k = np.mat(self.gradient(self.start))
    #         p_k = - 1.0 * np.mat(H_k) * g_k
    #         # p_k = mat(-linalg.solve(B_k, g_k)) # search direction
    #         # alpha_k = LineSearch(x0=self.start, xs=self.xs, ys=self.ys, r_func=self.r_func).evaluate()
    #         # alpha_k = wolfe(self.start, p_k)
    #         x_k_old = self.start.copy()
    #         self.start += p_k * alpha_k
    #         g_k_old = g_k
    #         g_k = np.mat(self.gradient(self.start))
    #         s_k = self.start - x_k_old
    #         y_k = g_k - g_k_old
    #
    #         if s_k.T * y_k > 0:
    #             H_k = (I - (s_k * y_k.T) / (y_k.T * s_k)) * H_k * (I - (y_k * s_k.T) / (y_k.T * s_k)) + s_k * s_k.T / (
    #                         y_k.T * s_k)
    #             # B_k = B_k - 1.0 * (B_k * s_k * s_k.T * B_k) / (s_k.T * B_k * s_k)\
    #             #       + 1.0 * (y_k * y_k.T) / (s_k.T * y_k)
    #
    #         k += 1
    #         y_array.append(self.f(self.start))
    #         print(k)
    #
    #     return self.start

def main():
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    data = DatasetReader('planar').parse()
    X, Y = np.array(data.input)[:, 0], np.array(data.output)
    result = bfgs_method(np.ones(10), X, Y)

    # X = np.linspace(1, 5, 50)
    # Y = f([2, 3], X) + np.random.normal(0, 0.1, size=50)
    # result = bfgs_method([10, 10], X, Y)

    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(X, Y, show_image=True)


if __name__ == '__main__':
    main()
