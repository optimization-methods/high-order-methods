from math import sqrt

import numpy as np
import numpy.linalg as ln
from matplotlib import pyplot as plt, cm
import matplotlib as mpl

from utils.drawer import SGDResult, Drawer

mpl.use('TkAgg')


def Jacobian(m_f, b, x):
    eps = 1e-6
    grads = []
    for i in range(len(b)):
        t = np.zeros(len(b)).astype(float)
        t[i] = t[i] + eps
        grad = (m_f(b + t, x) - m_f(b - t, x)) / (2 * eps)
        grads.append(grad)
    '''
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))`
    >>> np.column_stack((a,b))
    array([[1, 2],
           [2, 3],
           [3, 4]])
    '''
    return np.column_stack(grads)


def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# Gradient
def jac(x):
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 + 2 * x[0], 200 * x[1] - 200 * x[0] ** 2])


# Hessian
def hess(x):
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])


def dogleg_method(g, H, trust_radius):
    # Compute the Newton point.
    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    B = -np.dot(np.linalg.inv(H), g)

    # Test if the full step is within the trust region.
    if sqrt(np.dot(B, B)) <= trust_radius:
        return B

    # Compute the Cauchy point.
    # This is the predicted optimum along the direction of steepest descent.
    A = - (np.dot(g, g) / np.dot(g, np.dot(H, g))) * g

    dot_A = np.dot(A, A)
    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    if sqrt(dot_A) >= trust_radius:
        return trust_radius * A / sqrt(dot_A)

    # Find the solution to the scalar quadratic equation.
    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + tau*(p_b - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    V = B - A
    dot_V = np.dot(V, V)
    dot_A_V = np.dot(A, V)
    fact = dot_A_V ** 2 - dot_V * (dot_A - trust_radius ** 2)
    tau = (-dot_A_V + sqrt(fact)) / dot_V

    # Decide on which part of the trajectory to take.
    return A + tau * (B - A)


def trust_region_dogleg(func, x0, initial_trust_radius=1.0, eta=0.15, epoch=100):
    tol = 1e-4
    max_trust_radius = 100.0

    points = []

    point = x0
    points.append(x0)

    trust_radius = initial_trust_radius
    for i in range(epoch):
        gk = jac(point)
        Bk = hess(point)

        direction = dogleg_method(gk, Bk, trust_radius)

        act_red = func(point) - func(point + direction)
        pred_red = -(np.dot(gk, direction) + 0.5 * np.dot(direction, np.dot(Bk, direction)))
        rhok = act_red / pred_red
        # if pred_red == 0.0:
        #     rhok = 1e99
        # else:
        #     rhok = act_red / pred_red

        norm_pk = sqrt(np.dot(direction, direction))
        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else:
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        if rhok > eta:
            point = point + direction
        else:
            point = point

        points.append(point.tolist())

        if ln.norm(gk) < tol:
            break

    # return points
    return SGDResult(points, points, func, method_name='Dogleg')


def main():
    # np.set_printoptions(suppress=True)
    # result = np.array([arr.tolist() for arr in trust_region_dogleg(f, [5, 5])])[::4]

    result = trust_region_dogleg(f, [5, 5])
    drawer = Drawer(result)
    drawer.draw_3d(show_image=True)

    # last_point = result[-1]
    # shift, amount = 10, 1000
    # X1 = np.linspace(last_point[0] - shift, last_point[0] + shift, amount)
    # X2 = np.linspace(last_point[1] - shift, last_point[1] + shift, amount)
    # X1, X2 = np.meshgrid(X1, X2)
    # X = np.column_stack([X1.ravel(), X2.ravel()])
    # Z = np.array([f(x) for x in X])
    #
    # ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
    # ax.plot_surface(X1, X2, Z.reshape(amount, amount),
    #                 alpha=0.2,
    #                 color='green')
    #
    # levels = sorted(list(set([f(p) for p in result])))
    # ax.contour(X1, X2, Z.reshape(amount, amount), levels=levels,
    #            cmap='rainbow',
    #            alpha=0.8,
    #            linestyles='dashed',
    #            linewidths=0.5)
    # ax.plot(result[:, 0], result[:, 1], [f(p) for p in result],
    #         alpha=1, marker='o', markersize=3, linestyle='solid', linewidth=2)
    # colors = iter(cm.hot(np.flip(np.linspace(0, 1, len(result)))))
    # for i in range(1, len(result)):
    #     color = next(colors)
    #     ax.plot([result[i - 1][0], result[i][0]],
    #             [result[i - 1][1], result[i][1]],
    #             [f(result[i - 1]), f(result[i])],
    #             alpha=1, color=color, marker='o', markersize=3, linestyle='solid', linewidth=2)
    # plt.show()


if __name__ == '__main__':
    main()
