import numpy as np
from matplotlib import pyplot as plt


class BFGS:
    def __init__(self, r, xs, ys):
        self.r = r
        self.xs = xs
        self.ys = ys

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

    def fun(self, x_k):
        return np.sum(np.square(self.dy(x_k)))

    # grad fun
    def fun_grad(self, x_k):
        dy = self.dy(x_k)
        jacobian = self.jacobian(x_k)
        return [np.stack([elem]) for elem in -2 * jacobian.T @ dy]

    def wolfe(self, x_k, p_k):
        alpha = 1
        c1 = 1e-4
        c2 = 0.9
        rho = 0.8

        k = 0
        maxIt = 20
        # wolfe condition
        while ((k < maxIt) and (
                (self.fun(x_k + alpha * p_k) > self.fun(x_k) + c1 * (alpha * self.fun_grad(x_k).T * p_k)) or (
                (self.fun_grad(x_k + alpha * p_k).T * p_k) < c2 * (self.fun_grad(x_k).T * p_k)))):  # condition 2
            alpha *= rho
            k += 1
        return alpha

    def converge(self, x_k, eps=1e-4):
        m = np.shape(x_k)[0]
        # B_k = eye(m)
        H_k = np.eye(m)
        I = np.eye(m)
        y_array = [self.fun(x_k)]
        k = 1
        while abs(self.fun_grad(x_k)[0]) > eps:
            g_k = np.mat(self.fun_grad(x_k))
            print(f'{g_k=}')
            p_k = - 1.0 * np.mat(H_k) * g_k
            alpha_k = self.wolfe(x_k, p_k)
            x_k_old = x_k.copy()
            x_k += p_k * alpha_k
            g_k_old = g_k
            g_k = np.mat(self.fun_grad(x_k))
            s_k = x_k - x_k_old
            y_k = g_k - g_k_old

            if s_k.T * y_k > 0:
                H_k = (I - (s_k * y_k.T) / (y_k.T * s_k)) * H_k * (I - (y_k * s_k.T) / (y_k.T * s_k)) + s_k * s_k.T / (
                        y_k.T * s_k)
                # B_k = B_k - 1.0 * (B_k * s_k * s_k.T * B_k) / (s_k.T * B_k * s_k)\
                #       + 1.0 * (y_k * y_k.T) / (s_k.T * y_k)

            k += 1
            y_array.append(self.fun(x_k))
            print(k)

        plt.plot(y_array, 'g*-')
        plt.show()

        return x_k


if __name__ == "__main__":
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
    result = BFGS(r, xs, ys).converge(np.mat([[0.], [0.]]))
    print(result)

    # drawer = Drawer(result)
    # drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)
