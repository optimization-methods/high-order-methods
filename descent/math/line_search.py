import numpy as np


class LineSearch:
    def __init__(self, point, d, function, gradient):
        self.point = point
        self.d = d
        self.function = function
        self.gradient = gradient

    def __g(self, alpha):
        return self.function(self.point + alpha * self.d)

    def __grad(self, alpha):
        return self.gradient(self.point + alpha * self.d)

    def __spec_grad(self, alpha):
        return np.dot(self.__grad(alpha).T, self.d)

    def __data(self, alpha):
        return self.__g(alpha), self.__grad(alpha).T @ self.d

    def lowering(self, epoch=20, alpha=1, c1=1e-4, c2=0.9, rho=0.8):
        k = 0
        g_0, gamma_0 = self.__data(0)
        while True:
            g_alpha, gamma_alpha = self.__data(alpha)
            first = (g_alpha > g_0 + c1 * (alpha * gamma_0))
            second = (gamma_alpha < c2 * gamma_0)
            if (k < epoch) and (first or second):
                break
            alpha *= rho
            k += 1
        return alpha

    def find(self, max_iter=100, c1=10 ** (-3), c2=0.9, alpha_1=1.0, alpha_max=10 ** 6):
        if alpha_1 >= alpha_max:
            raise ValueError('Argument alpha_1 should be less than alpha_max')

        alpha_old = 0
        alpha_new = alpha_1

        final_alpha = None

        for i in np.arange(1, max_iter + 1):
            g_alpha = self.__g(alpha_new)
            if (i == 1 and g_alpha > self.__g(0) + c1 * alpha_new * self.__spec_grad(0)) or (
                    i > 1 and g_alpha >= self.__g(alpha_old)):
                final_alpha = self.__zoom(alpha_old, alpha_new, c1, c2)
                break

            g_spec_grad_alpha = self.__spec_grad(alpha_new)

            if np.abs(g_spec_grad_alpha) <= -c2 * self.__spec_grad(0):
                final_alpha = alpha_new
                break

            if g_spec_grad_alpha >= 0:
                final_alpha = self.__zoom(alpha_new, alpha_old, c1, c2)
                break

            alpha_old = alpha_new
            alpha_new = alpha_new + (alpha_max - alpha_new) * np.random.rand(1)

            if i == max_iter and final_alpha is None:
                return None

        return final_alpha

    def __zoom(self, alpha_lo, alpha_hi, c1, c2):
        while True:
            alpha_j = (alpha_hi + alpha_lo) / 2

            g_alpha_j = self.__g(alpha_j)
            g_0 = self.__g(0)

            if (g_alpha_j > g_0 + c1 * alpha_j * self.__spec_grad(0)) or (g_alpha_j >= self.__g(alpha_lo)):
                alpha_hi = alpha_j
            else:
                g_spec_grad_alpha_j = self.__spec_grad(alpha_j)

                if np.abs(g_spec_grad_alpha_j) <= -c2 * self.__spec_grad(0):
                    return alpha_j

                if g_spec_grad_alpha_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha_j
