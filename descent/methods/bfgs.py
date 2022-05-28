
import numpy as np
import numpy.linalg as ln

from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer


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
        H = I.copy()
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

        return DescentResult(self.f, points, points, r=self.r, method_name='BFGS')


def main():
    def r(m_b, m_x):
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    # def r(m_b, m_x):
    #     return m_b[0] * m_x / (m_b[1] + m_x)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    data = DatasetReader('planar').parse()
    xs, ys = np.array(data.input)[:, 0], np.array(data.output)
    result = BfgsDescentMethod(r, np.ones(5), xs, ys).converge()

    # xs = np.linspace(1, 5, 50)
    # ys = r([2, 3], xs) + np.random.normal(0, 0.1, size=50)
    # result = BfgsDescentMethod(r, [10, 10], xs, ys).converge()
    # print(result.scaled_scalars[-1])

    drawer = Drawer(result)
    drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


if __name__ == '__main__':
    main()
