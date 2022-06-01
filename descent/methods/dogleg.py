from math import sqrt

import numpy as np
import numpy.linalg as ln

from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult

# def f(m_b, m_x):
#     return m_b[0] * m_x / (m_b[1] + m_x)


# def f(m_b, m_x):
#     accumulator = 0
#     for i in range(len(m_b)):
#         accumulator += m_b[i] * m_x ** i
#     return accumulator


# def f(m_b, m_x):
#     return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2


# def get_jacobian(b, x):
#     eps = 1e-6
#     grads = []
#     for i in range(len(b)):
#         t = np.zeros(len(b)).astype(float)
#         t[i] = t[i] + eps
#         grad = (f(b + t, x) - f(b - t, x)) / (2 * eps)
#         grads.append(grad)
#     return np.column_stack(grads)


# def trust_region_dogleg(x0, X, Y, initial_trust_radius=1.0, eta=0.15, epoch=30):
#     tol = 1e-4
#     max_trust_radius = 100.0
#
#     points = []
#
#     point = x0
#     points.append(x0)
#
#     trust_radius = initial_trust_radius
#     for i in range(epoch):
#         dy = Y - f(point, X)
#         J = Jacobian(point, X)
#         g = - 2 * J.T @ dy
#         H = 2 * J.T @ J
#
#         direction = dogleg_method(g, H, trust_radius)
#
#         new_dy = Y - f(point + direction, X)
#
#         act_red = np.sum(dy ** 2) - np.sum(new_dy ** 2)
#         pred_red = -(np.dot(g, direction) + 0.5 * np.dot(direction, np.dot(H, direction)))
#
#         rhok = act_red / pred_red
#         if pred_red == 0.0:
#             rhok = 1e99
#
#         norm_pk = sqrt(np.dot(direction, direction))
#         if rhok < 0.25:
#             trust_radius = 0.25 * norm_pk
#         else:
#             if rhok > 0.75 and norm_pk == trust_radius:
#                 trust_radius = min(2.0 * trust_radius, max_trust_radius)
#             else:
#                 trust_radius = trust_radius
#
#         if rhok > eta:
#             point = point + direction
#         else:
#             point = point
#
#         points.append(point.tolist())
#         if ln.norm(g) < tol:
#             break
#
#     # return point
#     print(points[-1])
#     return DescentResult(points, points, f, method_name='Dogleg')


# def test(coefficients, start, stop, size):
#     data = DatasetReader('planar').parse()
#     X, Y = np.array(data.input)[:, 0], np.array(data.output)
#     result = trust_region_dogleg(coefficients, X, Y)
#
#     # X = np.linspace(start, stop, size)
#     # Y = f([2, 3], X) + np.random.normal(0, 0.1, size=size)
#     # result = trust_region_dogleg(coefficients, X, Y)
#
#     # X1 = np.linspace(start, stop, size)
#     # X2 = np.linspace(start, stop, size)
#     # X1, X2 = np.meshgrid(X1, X2)
#     # X = np.column_stack([X1.ravel(), X2.ravel()])
#     # Y = f([2, 3, 1], X) + np.random.normal(0, 1, size=len(X))
#     # result = trust_region_dogleg(coefficients, X, Y)
#
#     drawer = Drawer(result)
#     drawer.draw_2d_nonlinear_regression(X, Y, show_image=True)
#     # drawer.draw_3d_nonlinear_regression(X1, X2, Y, show_image=True)


# def main():
#     np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#     test(np.ones(10), -5, 5, 100)
#
#
# if __name__ == '__main__':
#     main()
from utils import config


class DoglegDescentMethod(DescentMethod):
    def __init__(self, func, start, xs, ys,
                 tol=1e-4,
                 initial_trust_radius=1.0,
                 max_trust_radius=100.0,
                 eta=0.15,
                 epoch=30):
        self.r = func
        self.start = start
        self.xs = np.array(xs, dtype=config.dtype)
        self.ys = np.array(ys, dtype=config.dtype)

        self.tol = tol
        self.initial_trust_radius = initial_trust_radius
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.epoch = epoch

    def jacobian(self, b, x):
        eps = 1e-6
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r(b + t, x) - self.r(b - t, x)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    @staticmethod
    def dogleg_method(g, H, trust_radius):
        B = -np.dot(np.linalg.inv(H), g)

        if sqrt(np.dot(B, B)) <= trust_radius:
            return B

        A = - (np.dot(g, g) / np.dot(g, np.dot(H, g))) * g

        dot_A = np.dot(A, A)
        if sqrt(dot_A) >= trust_radius:
            return trust_radius * A / sqrt(dot_A)

        V = B - A
        Dot_V = np.dot(V, V)
        dot_A_V = np.dot(A, V)
        fact = dot_A_V ** 2 - Dot_V * (dot_A - trust_radius ** 2)
        tau = (-dot_A_V + sqrt(fact)) / Dot_V

        return A + tau * V

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def dy(self, b):
        return self.ys - self.r(b, self.xs)

    def converge(self):
        points = []

        point = self.start
        points.append(self.start)

        trust_radius = self.initial_trust_radius
        for i in range(self.epoch):
            dy = self.dy(point)
            jacobian = self.jacobian(point, self.xs)
            g = -2 * jacobian.T @ dy
            hessian = 2 * jacobian.T @ jacobian

            direction = self.dogleg_method(g, hessian, trust_radius)

            new_dy = self.dy(point + direction)

            act_red = np.sum(dy ** 2) - np.sum(new_dy ** 2)
            pred_red = -(np.dot(g, direction) + 0.5 * np.dot(direction, np.dot(hessian, direction)))

            rhok = act_red / pred_red
            if pred_red == 0.0:
                rhok = 1e99

            norm_pk = sqrt(np.dot(direction, direction))
            if rhok < 0.25:
                trust_radius = 0.25 * norm_pk
            else:
                if rhok > 0.75 and norm_pk == trust_radius:
                    trust_radius = min(2.0 * trust_radius, self.max_trust_radius)
                else:
                    trust_radius = trust_radius

            point = point + direction if rhok > self.eta else point

            points.append(point.tolist())
            if ln.norm(g) < self.tol:
                break

        return DescentResult(self.f, points, points, r=self.r, method_name='Dogleg')

    @property
    def name(self):
        return 'Dogleg'
