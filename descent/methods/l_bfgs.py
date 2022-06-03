from collections import deque

import numpy as np

from descent.math.calculus import Calculus
from descent.math.line_search import LineSearch
from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config


class LBFGSDescentMethod(DescentMethod):
    def __init__(self, r, start, xs, ys, eps=1e-4, max_iter=100, m=5):
        super().__init__(xs, ys, r)
        self.start = start
        self.calculus = Calculus(xs, ys, r)

        self.eps = eps
        self.max_iter = max_iter
        self.m = m

    def converge(self):
        point = self.start
        s = deque()
        y = deque()
        points = [point]

        eye = np.eye(len(point), dtype=config.dtype)
        h_0 = eye.copy()

        k = 0
        while np.linalg.norm(self.calculus.gradient(point)) > self.eps and k < self.max_iter:
            if k > 0:
                gamma = (s[-1].T @ y[-1]) / (y[-1].T @ y[-1])
                h_0 = gamma * eye

            direction = self.calc_direction(h_0, point, s, y)
            alpha = LineSearch(point, direction, self.calculus.f, self.calculus.gradient).find()

            point_old = point.copy()
            point = point + alpha * np.array(direction, dtype=config.dtype)

            if k >= self.m:
                s.popleft()
                y.popleft()

            s.append(point - point_old)
            y.append(self.calculus.gradient(point) - self.calculus.gradient(point_old))

            points.append(point.tolist())
            k += 1
            # print(k)

        # noinspection SpellCheckingInspection
        return DescentResult(self.calculus.f, points, points, r=self._r, method_name='L_BFGS')

    def calc_direction(self, h_0, point, s, y):
        q = self.calculus.gradient(point)
        alpha_arr = []

        t = len(s)
        for i in range(t):
            alpha = (s[t - i - 1].T @ q) / (y[t - i - 1].T @ s[t - i - 1])
            q = q - alpha * y[t - i - 1]
            alpha_arr.append(alpha)

        r = np.dot(h_0, q)
        for i in range(t):
            beta = (y[i].T @ r) / (y[i].T @ s[i])
            r += s[i] * (alpha_arr[t - i - 1] - beta)

        return -r

    @property
    def name(self):
        # noinspection SpellCheckingInspection
        return 'L-BFGS'
