from math import sqrt

import numpy as np
import numpy.linalg as ln

from descent.math.calculus import Calculus
from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult


class DoglegDescentMethod(DescentMethod):
    def __init__(self,
                 r, start, xs, ys,
                 tol=1e-4,
                 initial_trust_radius=1.0,
                 max_trust_radius=100.0,
                 eta=0.15,
                 epoch=30):
        super().__init__(xs, ys, r)
        self.start = start
        self.calculus = Calculus(xs, ys, r)

        self.tol = tol
        self.initial_trust_radius = initial_trust_radius
        self.max_trust_radius = max_trust_radius
        self.eta = eta
        self.epoch = epoch

    @staticmethod
    def dogleg_method(g, h, trust_radius):
        b = -np.dot(np.linalg.inv(h), g)

        if sqrt(np.dot(b, b)) <= trust_radius:
            return b

        a = - (np.dot(g, g) / np.dot(g, np.dot(h, g))) * g

        dot_a = np.dot(a, a)
        if sqrt(dot_a) >= trust_radius:
            return trust_radius * a / sqrt(dot_a)

        v = b - a
        dot_v_v = np.dot(v, v)
        dot_a_v = np.dot(a, v)
        fact = dot_a_v ** 2 - dot_v_v * (dot_a - trust_radius ** 2)
        tau = (-dot_a_v + sqrt(fact)) / dot_v_v

        return a + tau * v

    def converge(self):
        points = []

        point = self.start
        points.append(self.start)

        trust_radius = self.initial_trust_radius
        for i in range(self.epoch):
            dy = self.calculus.dy(point)
            jacobian = self.calculus.jacobian(point)
            g = -2 * jacobian.T @ dy
            hessian = 2 * jacobian.T @ jacobian

            direction = self.dogleg_method(g, hessian, trust_radius)

            new_dy = self.dy(point + direction)

            act_red = np.sum(dy ** 2) - np.sum(new_dy ** 2)
            pred_red = -(np.dot(g, direction) + 0.5 * np.dot(direction, np.dot(hessian, direction)))

            rho = act_red / pred_red if pred_red != 0.0 else 1e99

            norm_direction = sqrt(np.dot(direction, direction))
            if rho < 0.25:
                trust_radius = 0.25 * norm_direction
            else:
                if rho > 0.75 and norm_direction == trust_radius:
                    trust_radius = min(2.0 * trust_radius, self.max_trust_radius)
                else:
                    trust_radius = trust_radius

            point = point + direction if rho > self.eta else point

            points.append(point.tolist())
            if ln.norm(g) < self.tol:
                break

        return DescentResult(self.f, points, points, r=self.r, method_name='Dogleg')

    @property
    def name(self):
        return 'Dogleg'
