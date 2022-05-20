import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from descent.methods.descent_result import DescentResult
from descent.methods.line_search import grad, special_grad, g
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer

import matplotlib as mpl

mpl.use("TkAgg")


def r_func(m_b, m_x):
    accumulator = 0
    for i in range(len(m_b)):
        accumulator += m_b[i] * m_x ** i
    return accumulator


def f(m_b, m_x, m_y):
    return np.sum((m_y - r_func(m_b, m_x)) ** 2)


def Jacobian(b, x):
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros(len(b)).astype(float)
        t[i] = t[i] + eps
        grad = (r_func(b + t, x) - r_func(b - t, x)) / (2 * eps)
        grads.append(grad)
    return np.column_stack(grads)

def line_search(x0, X, Y, p, nabla):
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x0, X, Y)
    x_new = x0 + a * p
    nabla_new = grad(x_new, X, Y)
    jok = 0
    while f(x_new, X, Y) >= fx + (c1 * a * nabla.T @ p) or nabla_new.T @ p <= c2 * nabla.T @ p:
        a *= 0.99
        x_new = x0 + a * p
        nabla_new = grad(x_new, X, Y)
        if jok % 100 == 0:
            print("jok", jok)
        jok += 1
    return a


def BFGS(x0, X, Y, max_it, plot=False):
    d = len(x0)  # dimension of problem
    nabla = grad(x0, X, Y)  # initial gradient
    H = np.eye(d)  # initial hessian
    x = x0[:]
    it = 2
    if plot:
        if d == 2:
            x_store = np.zeros((1, 2))  # storing x values
            x_store[0, :] = x
        else:
            x_store = np.zeros((1, d))
            np.append(x_store, x)
            print('Too many dimensions to produce trajectory plot!')
            plot = False

    while np.linalg.norm(nabla) > 1e-5:  # while gradient is positive
        if it > max_it:
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H @ nabla  # search direction (Newton Method) pigs the color of the object in the image
        # a = line_search(x, X, Y, p, nabla)  # line search pigment of x value in the image of the original image
        a = scipy.optimize.line_search(lambda xx: f(xx, X, Y), lambda xx: grad(xx, X, Y), x0, p)
        for jok in a:
            if jok is not None:
                a = jok
                break
        # print(a, p)
        s = a * p
        x_new = x + a * p
        nabla_new = grad(x_new, X, Y)
        y = nabla_new - nabla
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y, (d, 1))
        s = np.reshape(s, (d, 1))
        r = 1 / (y.T @ s)
        li = (np.eye(d) - (r * (s @ y.T)))
        ri = (np.eye(d) - (r * (y @ s.T)))
        hess_inter = li @ H @ ri
        H = hess_inter + (r * (s @ s.T))  # BFGS Update
        nabla = nabla_new[:]
        x = x_new[:]
        # if plot:
        x_store = np.append(x_store, [x], axis=0)  # storing x
    if plot:
        x1 = np.linspace(min(x_store[:, 0] - 0.5), max(x_store[:, 0] + 0.5), 30)
        x2 = np.linspace(min(x_store[:, 1] - 0.5), max(x_store[:, 1] + 0.5), 30)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.ravel(), X2.ravel()])
        Z = f(x, X).reshape(len(x1), len(x2))
        plt.figure()
        plt.title('OPTIMAL AT: ' + str(x_store[-1, :]) + '\n IN ' + str(len(x_store)) + ' ITERATIONS')
        plt.contourf(X1, X2, Z, 30, cmap='jet')
        plt.colorbar()
        plt.plot(x_store[:, 0], x_store[:, 1], c='w')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.show()
    return DescentResult(x_store, x_store, r_func, 'jok')


data = DatasetReader('planar').parse()
xs = np.array(data.input)[:, 0]
ys = data.output
x_opt = BFGS(np.ones(10), xs, ys, 100, plot=True)
print(x_opt)
drawer = Drawer(x_opt)
drawer.draw_2d_nonlinear_regression(xs, ys, show_image=True)
