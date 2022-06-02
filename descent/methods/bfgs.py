import numpy as np
import numpy.linalg as ln

from descent.math.calculus import Calculus
from descent.math.line_search import LineSearch
from descent.methods.descent_method import DescentMethod
from descent.methods.descent_result import DescentResult
from utils import config
from utils.dataset_reader import DatasetReader
from utils.drawer import Drawer



class BFGSDescentMethod(DescentMethod):
    def __init__(self, r, start, xs, ys):
        super().__init__(xs, ys, r)
        self.start = start
        self.calculus = Calculus(xs, ys, r)

        self.eps = 10e-3

    def gradient(self, x):
        dy = self.dy(x)
        jacobian = self.jacobian(x)
        return -2 * jacobian.T @ dy

    def converge(self):
        g = self.gradient(self.start)
        eye = np.eye(len(self.start), dtype=config.dtype)
        h = eye.copy()
        points = [self.start]
        x0 = points[-1]

        while ln.norm(g) > self.eps:
            direction = -np.dot(h, g)

            alpha = LineSearch(x0, direction,
                               function=self.calculus.f,
                               gradient=self.gradient
                               ).find()

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

        # noinspection SpellCheckingInspection
        return DescentResult(self.calculus.f, points, points, r=self._r, method_name='BFGS')

    @property
    def name(self):
        # noinspection SpellCheckingInspection
        return 'BFGS'
