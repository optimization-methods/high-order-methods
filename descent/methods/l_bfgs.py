# noinspection SpellCheckingInspection
import numpy as np

from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


# noinspection SpellCheckingInspection
class LBfgsDescentMethod(DescentMethod):
    def __init__(self, r_func, start, xs, ys):
        self.start = start
        self.r_func = r_func
        self.ys = ys
        self.xs = xs

    def jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r_func(b + t, self.xs) - self.r_func(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    @staticmethod
    def r(b, x):
        accumulator = 0
        for i in range(len(b)):
            accumulator += 1 / b[i] * x ** i
        return accumulator

    def dy(self, b):
        return self.ys - self.r_func(b, self.xs)

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def grad(self, b):
        dy = self.dy(b)
        jacobian = self.jacobian(b)
        return -2 * jacobian.T @ dy

    def converge(self, eps=1e-4):
        point = self.start

        y_array = [self.f(point)]
        H_0 = np.eye(np.shape(point)[0])
        I = np.eye(np.shape(point)[0])
        m = 3
        s = y = []
        points = [point]

        g_k = self.grad(point)
        p_k = -H_0 * g_k

        k = 0
        while abs(self.grad(point)[0]) > eps:
            if k > 0:
                a = np.dot(np.transpose(s[-1]), y[-1])
                b = np.dot(np.transpose(y[-1]), y[-1])
                H_0 = I * (a / b)

            t = len(s)
            q_k = self.grad(point)
            alpha_i = []
            for i in range(t):
                a = np.dot(np.transpose(s[t - i - 1]), q_k)
                b = np.dot(np.transpose(y[t - i - 1]), s[t - i - 1])
                alpha = a / b
                q_k = q_k - alpha * y[t - i - 1]
                alpha_i.append(alpha)

            gradi = np.dot(H_0, q_k)

            for i in range(t):
                a = np.dot(np.transpose(y[i]), gradi)
                b = np.dot(np.transpose(y[i]), s[i])
                gradi += np.dot(s[i], (alpha_i[t - i - 1] - (a / b)))

            p_k = -gradi

            alpha = self.wolfe(point, p_k)
            point_old = point.copy()
            point = point + alpha * p_k

            if k >= m:
                s.pop(0)
                y.pop(0)

            s.append(point - point_old)
            y.append(self.grad(point) - self.grad(point_old))
            y_array.append(self.f(point))

            points.append(point)

            k += 1

        return DescentResult(np.array(points), np.array(points), self.r_func, 'L_BFGS')

    def wolfe(self, x_k, p_k):
        x_k = np.array(x_k, dtype=config.dtype)

        alpha = 1
        c1 = 1e-4
        c2 = 0.9
        rho = 0.8

        k = 0
        max_it = 20
        while ((k < max_it) and (
                (
                        self.f(x_k + alpha * p_k) >
                        self.f(x_k) + c1 * (alpha * x_k.T @ p_k)
                ) or
                (
                        self.grad(x_k + alpha * p_k).T @ p_k <
                        c2 * self.grad(x_k).T @ p_k
                )
        )):
            alpha *= rho
            k += 1
        return alpha


if __name__ == "__main__":
    # def r(m_b, m_x):
    #     return m_b[0] * m_x / (m_b[1] + m_x)

    def r(b, x):
        accumulator = 0
        for i in range(len(b)):
            accumulator += 1 / b[i] * x ** i
        return accumulator

    # xs = np.linspace(1, 5, 50)
    # ys = r([1, 1], xs) + np.random.normal(0, 1, size=50)
    # x_0 = [5, 5]
    #
    data = DatasetReader('planar').parse()
    xs, ys = np.array(data.input)[:, 0], np.array(data.output)
    x_0 = np.ones(10)

    # x_0 = np.mat([[0.], [0.]])
    result = LBfgsDescentMethod(x_0, r, xs, ys).converge()

    print(result)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)

