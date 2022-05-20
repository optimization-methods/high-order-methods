import numpy as np


class LineSearch:
    def __init__(self, x0, xs, ys, r_func):
        self.x0 = x0
        self.xs = xs
        self.ys = ys
        self.r_func = r_func
        # print("JOK")
        # print(-self.derivative(self.x0))
        # print("COK")
        # self.der = 1
        # self.der = -self.derivative(self.x0)

    def dy(self, m_b):
        return self.ys - self.r_func(m_b, self.xs)

    def f(self, m_b):
        return np.sum(self.dy(m_b) ** 2)

    def Jacobian(self, b):
        eps = 1e-6
        grads = []
        for i in range(len(b)):
            t = np.zeros(len(b)).astype(float)
            t[i] = t[i] + eps
            grad = (self.r_func(b + t, self.xs) - self.r_func(b - t, self.xs)) / (2 * eps)
            grads.append(grad)
        return np.column_stack(grads)

    def derivative(self, alpha, h=1e-5):
        from scipy import misc
        return misc.derivative(self.f, alpha)
        # return (self.g(alpha + h) - self.g(alpha - h)) / (2 * h)

    def grad(self, b):
        dy = self.dy(b)
        jacobian = self.Jacobian(b)
        return -2 * jacobian.T @ dy

    def g(self, alpha):
        return self.f(self.x0 + alpha * self.der)

    # def evaluate(self):
    #     alpha_0 = 0
    #     alpha_max = 1
    #     alpha_1 = 0.7
    #
    #     c1 = 1e-4  # c1: Armijo condition
    #     c2 = 0.9  # c2: curvature condition
    #
    #     alpha_pre = alpha_0
    #     alpha_cur = alpha_1
    #     alpha_min = 1e-7
    #
    #     i = 0
    #     eps = 1e-16
    #     while abs(alpha_cur - alpha_pre) >= eps:
    #         phi_alpha_cur = self.f(self.x0 + alpha_cur * -(self.derivative(self.x0)))
    #         phi_alpha_pre = self.f(self.x0 + alpha_pre * -(self.derivative(self.x0)))
    #         phi_alpha_0 = self.f(self.x0)
    #         phi_grad_alpha_0 = self.f(self.x0) * (-self.derivative(self.x0))
    #
    #         if phi_alpha_cur > phi_alpha_0 + c1 * alpha_cur * phi_grad_alpha_0 or (
    #                 phi_alpha_cur > phi_alpha_pre and i > 0):
    #             return self.zoom(alpha_pre, alpha_cur)
    #
    #         phi_grad_alpha_cur = self.f(self.x0 + alpha_cur * (-self.derivative(self.x0))) * (-self.derivative(self.x0))
    #         if abs(phi_grad_alpha_cur) <= -c2 * phi_grad_alpha_0:  # satisfy Wolfe condition
    #             return alpha_cur
    #
    #         if phi_grad_alpha_cur >= 0:
    #             return self.zoom(alpha_cur, alpha_max)
    #
    #         alpha_new = self.QuadraticInterpolation(alpha_cur, phi_alpha_cur, phi_alpha_0, phi_grad_alpha_0)
    #         alpha_pre = alpha_cur
    #         alpha_cur = alpha_new
    #         i += 1
    #
    #     return alpha_min
    #
    # @staticmethod
    # def QuadraticInterpolation(alpha, phi, phi0, g0):
    #     numerator = g0 * alpha * alpha
    #     denominator = -2 * (phi - g0 * alpha - phi0)
    #     return numerator / denominator
    #
    # def zoom(self, alpha_low, alpha_high):
    #     if alpha_low > alpha_high:
    #         raise ValueError(f'Alpha low must be lower than Alpha high: {alpha_low=}, {alpha_high=}')
    #
    #     c1 = 1e-4  # c1: Armijo condition
    #     c2 = 0.9  # c2: curvature condition
    #
    #     eps = 1e-16
    #     while abs(alpha_high - alpha_low) >= eps:
    #         alpha_j = (alpha_low + alpha_high) / 2
    #         phi_alpha_j = self.f(self.x0 + alpha_j * (-self.derivative(self.x0)))  # direction: here select steepest descent
    #         phi_alpha_0 = self.f(self.x0)
    #         phi_alpha_low = self.f(self.x0 + alpha_low * (-self.derivative(self.x0)))
    #         phi_grad_alpha_0 = self.f(self.x0) * self.derivative(self.x0)
    #
    #         if phi_alpha_j > phi_alpha_0 + c1 * alpha_j * phi_grad_alpha_0 or phi_alpha_j >= phi_alpha_low:
    #             alpha_high = alpha_j
    #
    #         else:
    #             phi_grad_alpha_j = self.f(self.x0 + alpha_j * (-self.derivative(self.x0))) * self.derivative(self.x0)
    #
    #             if abs(phi_grad_alpha_j) <= -c2 * phi_grad_alpha_0:
    #                 return alpha_j
    #
    #             if phi_grad_alpha_j * (alpha_high - alpha_low) >= 0:
    #                 alpha_high = alpha_low
    #
    #             alpha_low = alpha_j
    #
    #     return alpha_low
    #
    # def zoom(self, alo, ahi, c1, c2):
    #     print('base', alo, ahi)
    #     fi_a0 = self.g(0)
    #     grad_fi_a0 = self.derivative(0)
    #     eps = 1e-16
    #     while abs(ahi - alo) >= eps:
    #         # print(alo, ahi)
    #         aj = (alo + ahi) / 2
    #         fi_aj = self.g(aj)
    #         fi_alo = self.g(alo)
    #         # print('jok', fi_aj, fi_a0 + c1 * aj * grad_fi_a0)
    #         boolpig = fi_aj > fi_a0 + c1 * aj * grad_fi_a0
    #         boolswine = fi_aj > fi_alo
    #         print(f'{boolpig=} {fi_aj=} {fi_alo=} {boolswine=}')
    #         if boolpig or boolswine:
    #             ahi = aj
    #         else:
    #             grad_fi_aj = self.derivative(aj)
    #             print(f'derivative of {aj} is {grad_fi_aj}')
    #             jok = abs(grad_fi_aj)
    #             cok = -c2 * grad_fi_a0
    #             print(jok, cok)
    #             if jok <= cok:
    #                 return aj
    #             if grad_fi_aj * (ahi - alo) >= 0:
    #                 ahi = alo
    #             alo = aj
    #     return alo
    #
    # def evaluate(self, a_max=100, c1=1e-4, c2=0.9):
    #     a = [0.0, a_max / 2.0]
    #     g_0 = self.g(a[0])
    #     grad_0 = self.derivative(a[0])
    #     i = 1
    #     while True:
    #         print("jok")
    #         g_curr = self.g(a[i])
    #         g_prev = self.g(a[i - 1])
    #         if g_curr > g_0 + c1 * a[i] * grad_0 or (g_curr >= g_prev and i > 1):
    #             return self.zoom(a[i - 1], a[i], c1, c2)
    #         grad_next = self.derivative(a[i])
    #         if abs(grad_next) <= - c2 * grad_0:
    #             return a[i]
    #         if grad_next >= 0:
    #             return self.zoom(a[i], a_max, c1, c2)
    #         a.append((a_max - a[i]) / 2)
    #         i = i + 1