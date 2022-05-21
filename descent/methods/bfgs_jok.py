import numpy as np

from utils import config
from utils.dataset_reader import DatasetReader


def line_search(fun, grad, x, p, maxiter=100, c1=10 ** (-3), c2=0.9, alpha_1=1.0, alpha_max=10 ** 6):
    if alpha_1 >= alpha_max:
        raise ValueError('Argument alpha_1 should be less than alpha_max')

    def phi(alpha):
        return fun(x + alpha * p)

    def phi_grad(alpha):
        return np.dot(grad(x + alpha * p).T, p)

    alpha_old = 0
    alpha_new = alpha_1

    final_alpha = None

    for i in np.arange(1, maxiter + 1):
        print('i love cocks')
        phi_alpha = phi(alpha_new)

        if (i == 1 and phi_alpha > phi(0) + c1 * alpha_new * phi_grad(0)) or (
                i > 1 and phi_alpha >= phi(alpha_old)):
            final_alpha = zoom(x, p, phi, phi_grad, alpha_old, alpha_new, c1, c2)
            break

        phi_grad_alpha = phi_grad(alpha_new)

        if np.abs(phi_grad_alpha) <= -c2 * phi_grad(0):
            final_alpha = alpha_new
            break

        if phi_grad_alpha >= 0:
            final_alpha = zoom(x, p, phi, phi_grad, alpha_new, alpha_old, c1, c2)
            break

        alpha_old = alpha_new
        alpha_new = alpha_new + (alpha_max - alpha_new) * np.random.rand(1)

    if i == maxiter and final_alpha is None:
        return None

    return final_alpha


def zoom(x, p, phi, phi_grad, alpha_lo, alpha_hi, c1, c2, eps=1e-22):
    # print(phi, phi_grad, alpha_lo, alpha_hi, c1, c2)
    # while abs(alpha_hi - alpha_lo) > eps:
    while True:
        alpha_j = (alpha_hi + alpha_lo) / 2
        print(f'piglet {alpha_lo=} {alpha_j=} {alpha_hi=}')

        phi_alpha_j = phi(alpha_j)

        if (phi_alpha_j > phi(0) + c1 * alpha_j * phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            print('first condition not satisfied')
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)

            jok = np.abs(phi_grad_alpha_j)
            cok = -c2 * phi_grad(0)
            print(f'{jok=} {cok=}')
            if jok <= cok:
                print('conditions satisfied')
                return alpha_j

            print('second condition not satisfied')

            if phi_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                print(f'pigs happened {alpha_hi=} {alpha_lo=}')
                alpha_hi = alpha_lo
                # alpha_lo, alpha_hi = alpha_hi, alpha_lo

            alpha_lo = alpha_j
    # return alpha_lo


class BFGS(object):
    def __init__(self, x0, xs, ys, r):
        self.x0 = x0
        self.xs = xs
        self.ys = ys
        self.r = r

        # amount = 1000
        # X1 = np.linspace(-5, 5, amount)
        # X2 = np.linspace(-5, 5, amount)
        # X1, X2 = np.meshgrid(X1, X2)
        # X = np.column_stack([X1.ravel(), X2.ravel()])
        # Z = self.f(X).reshape(amount, amount)
        # plt.plot(X1, X2, Z)

    def jacobian(self, b):
        eps = 1e-6
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r(b + t, self.xs) - self.r(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def dy(self, point):
        return self.ys - self.r(point, self.xs)

    def grad(self, point):
        dy = self.dy(point)
        jacobian = self.jacobian(point)
        return -2 * jacobian.T @ dy

    def f(self, point):
        return np.sum(np.array(self.dy(point), dtype=config.dtype) ** 2)

    def evaluate(self, eps=1e-4, max_iterations=100, verbose=False):
        n = len(self.x0)
        H_old = np.diag(np.ones(n))
        x_old = self.x0

        for i in np.arange(1, max_iterations + 1):
            print('jok')
            # Search direction
            p = -1 * np.dot(H_old, self.grad(x_old))

            alpha = line_search(self.f, self.grad, x_old, p, maxiter=max_iterations)

            if alpha is None:
                print('Wolfe line search did not converge')
                return x_old, i

            x_new = x_old + alpha * p

            s = (x_new - x_old).reshape((n, 1))
            y = (self.grad(x_new) - self.grad(x_old)).reshape((n, 1))
            sT = s.T.reshape((1, n))
            yT = y.T.reshape((1, n))

            yT_s = np.dot(yT, s).reshape(())

            I = np.diag(np.ones(n))
            rho = 1 / yT_s
            rho2 = rho ** 2

            H_y = np.dot(H_old, y).reshape((n, 1))
            Hy_sT = np.dot(H_y, sT).reshape((n, n))
            yT_H = np.dot(yT, H_old).reshape((1, n))
            s_yTH = np.dot(s, yT_H).reshape((n, n))
            syTH_y = np.dot(s_yTH, y).reshape((n, 1))
            syTHy_sT = np.dot(syTH_y, sT).reshape((n, n))
            s_sT = np.dot(s, sT).reshape((n, n))

            H_new = H_old - rho * Hy_sT - rho * s_yTH + rho2 * syTHy_sT + rho * s_sT

            if verbose:
                print('x_k = {0} converges to x_(k+1) = {1}'.format(x_old, x_new))

            grad_dist = np.linalg.norm(self.grad(x_old) - self.grad(x_new))
            if grad_dist < eps:
                break
            elif verbose:
                print('There is still {0} left for approximations to converge'.format(np.abs(grad_dist - eps)), '\n')

            x_old = x_new
            H_old = H_new

        if verbose:
            print('\nFinal approximation of the minima is {0}.'.format(x_new))
            if i != max_iterations:
                print('Optimization process converged in {0} steps'.format(i))
            else:
                print('Optimization process did not converge')

        return x_new, i


def main():
    def r(m_b, m_x):
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return np.exp(accumulator)

    # def r(m_b, m_x):
    #     return m_b[0] * m_x / (m_b[1] + m_x)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    data = DatasetReader('planar').parse()
    xs, ys = np.array(data.input)[:, 0], np.array(data.output)
    result = BFGS(np.ones(10), xs, ys, r).evaluate()
    print(result)

    # xs = np.linspace(1, 5, 50)
    # ys = r_func([2, 3], xs) + np.random.normal(0, 1, size=50)
    # result = BFGS([10, 10], xs, ys).evaluate()

    # drawer = Drawer(result)
    # drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)


if __name__ == '__main__':
    main()
