# noinspection SpellCheckingInspection
import numpy as np

from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


# noinspection SpellCheckingInspection
class LBfgsDescentMethod(DescentMethod):
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

        def spec_grad(self, alpha):
            return np.dot(self.grad(alpha).T, self.d)

        def data(self, alpha):
            return self.g(alpha), self.grad(alpha).T @ self.d

        def wolfe1(self, epoch=20, alpha=1, c1=1e-4, c2=0.9, rho=0.5):
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

        def wolfe(self, maxiter=100, c1=10**(-3), c2=0.9, alpha_1=1.0, alpha_max=10**6):
            if alpha_1 >= alpha_max:
                raise ValueError('Argument alpha_1 should be less than alpha_max')

            alpha_old = 0
            alpha_new = alpha_1

            final_alpha = None

            for i in np.arange(1, maxiter + 1):
                g_alpha = self.g(alpha_new)
                if (i == 1 and g_alpha > self.g(0) + c1 * alpha_new * self.spec_grad(0)) or (i > 1 and g_alpha >= self.g(alpha_old)):
                    final_alpha = self.zoom(alpha_old, alpha_new, c1, c2)
                    break

                g_spec_grad_alpha = self.spec_grad(alpha_new)

                if np.abs(g_spec_grad_alpha) <= -c2 * self.spec_grad(0):
                    final_alpha = alpha_new
                    break

                if g_spec_grad_alpha >= 0:
                    final_alpha = self.zoom(alpha_new, alpha_old, c1, c2)
                    break

                alpha_old = alpha_new
                alpha_new = alpha_new + (alpha_max - alpha_new) * np.random.rand(1)

                if i == maxiter and final_alpha is None:
                    return None

            return final_alpha

        def zoom(self, alpha_lo, alpha_hi, c1, c2):
            while True:
                alpha_j = (alpha_hi + alpha_lo) / 2

                g_alpha_j = self.g(alpha_j)
                g_0 = self.g(0)

                if (g_alpha_j > g_0 + c1 * alpha_j * self.spec_grad(0)) or (g_alpha_j >= self.g(alpha_lo)):
                    alpha_hi = alpha_j
                else:
                    g_spec_grad_alpha_j = self.spec_grad(alpha_j)

                    if np.abs(g_spec_grad_alpha_j) <= -c2 * self.spec_grad(0):
                        return alpha_j

                    if g_spec_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo

                    alpha_lo = alpha_j

    def __init__(self, r, start, xs, ys):
        self.start = start
        self.r = r
        self.ys = ys
        self.xs = xs

    def jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r(b + t, self.xs) - self.r(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def dy(self, b):
        return self.ys - self.r(b, self.xs)

    def f(self, b):
        return np.sum(self.dy(b) ** 2)

    def gradient(self, b):
        dy = self.dy(b)
        jacobian = self.jacobian(b)
        return -2 * jacobian.T @ dy

    def converge(self, eps=1e-4, maxiter=100):
        point = self.start

        y_array = [self.f(point)]
        H_0 = np.eye(np.shape(point)[0])
        I = np.eye(np.shape(point)[0])
        m = 3
        s = y = []
        points = [point]

        g_k = self.gradient(point)
        p_k = -H_0 * g_k

        k = 0
        while abs(self.gradient(point)[0]) > eps and k < maxiter:
            if k > 0:
                a = np.dot(np.transpose(s[-1]), y[-1])
                b = np.dot(np.transpose(y[-1]), y[-1])
                H_0 = I * (a / b)

            t = len(s)
            q_k = self.gradient(point)
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

            # alpha = self.wolfe(point, p_k)
            alpha = self.LineSearch(point, p_k, self).wolfe()
            point_old = point.copy()
            point = point + alpha * np.array(p_k)

            if k >= m:
                s.pop(0)
                y.pop(0)

            s.append(point - point_old)
            y.append(self.gradient(point) - self.gradient(point_old))
            y_array.append(self.f(point))

            points.append(point)
            # print(point)

            k += 1
            print(k)

        return DescentResult(np.array(points), np.array(points), self.r, 'L_BFGS')

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
                ) or (
                        self.gradient(x_k + alpha * p_k).T @ p_k <
                        c2 * self.gradient(x_k).T @ p_k
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
            accumulator += b[i] * x ** i
        return accumulator

    # xs = np.linspace(1, 5, 50)
    # ys = r([2, 3], xs) + np.random.normal(0, 0.1, size=50)
    # x_0 = [10, 10]

    data = DatasetReader('planar').parse()
    xs, ys = np.array(data.input)[:, 0], np.array(data.output)
    x_0 = np.ones(10)

    # x_0 = np.mat([[0.], [0.]])
    result = LBfgsDescentMethod(r, x_0, xs, ys).converge()

    print(result)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)

