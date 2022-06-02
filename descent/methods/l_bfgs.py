# noinspection SpellCheckingInspection
import numpy as np

from descent.math.line_search import LineSearch
from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


# noinspection SpellCheckingInspection
class LBFGSDescentMethod(DescentMethod):
    def __init__(self, r, start, xs, ys):
        super().__init__(xs, ys)
        self.start = start
        self.r = r

    def jacobian(self, b, eps=1e-6):
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b), dtype=config.dtype)
            t[i] = t[i] + eps
            grad = (self.r(b + t, self._xs) - self.r(b - t, self._xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def dy(self, b):
        return self._ys - self.r(b, self._xs)

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
            alpha = LineSearch(point, p_k, self).find()
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

        return DescentResult(self.f, np.array(points), np.array(points), r=self.r, method_name='L_BFGS')

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

    @property
    def name(self):
        return 'L-BFGS'


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
    result = LBFGSDescentMethod(r, x_0, xs, ys).converge()

    print(result)
    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)

