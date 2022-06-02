import os

import numpy as np
from matplotlib import pyplot as plt, cm
import utils.config as config

save_directory = f"{config.source}/images"


# noinspection SpellCheckingInspection
class Drawer(object):
    def __init__(self, sgd_result):
        self.scaled = np.array(sgd_result.scaled_scalars, dtype=config.dtype)
        self.rescaled = np.array(sgd_result.rescaled_scalars, dtype=config.dtype)
        self.func = sgd_result.func

        self.r = sgd_result.r

        self.batch_size = sgd_result.batch_size

        self.scaler_name = sgd_result.scaler_name
        self.method_name = sgd_result.method_name

        self.scatter_config = {
            'color': 'green',
            'alpha': 0.3
        }
        self.surface_config = {
            'alpha': 0.2,
            'color': 'green',
            'cmap': cm.coolwarm,
        }
        self.line_config = {
            'alpha': 1,
            'marker': 'o',
            'markersize': 3,
            'linestyle': 'solid',
            'linewidth': 2,
        }
        self.contour_config = {
            'cmap': 'autumn',
            'alpha': 0.8,
            'linestyles': 'dashed',
            'linewidths': 0.5,
        }

    def __calculate_values(self, amount, additional_shift):
        last_point = self.scaled[-1]
        start_point = self.scaled[0]

        x1_shift = abs(last_point[0] - start_point[0]) + additional_shift
        x2_shift = abs(last_point[1] - start_point[1]) + additional_shift

        x1 = np.linspace(
            last_point[0] - x1_shift,
            last_point[0] + x1_shift,
            amount
        )

        x2 = np.linspace(
            last_point[1] - x2_shift,
            last_point[1] + x2_shift,
            amount
        )

        x1, x2 = np.meshgrid(x1, x2)
        x = np.column_stack([x1.ravel(), x2.ravel()])
        z = np.array([self.func(p) for p in x], dtype=config.dtype).reshape(amount, amount)

        return x1, x2, z

    def draw_3d(self, show_image=True):
        """
        Requires a function already built from x and y.
        Minimum point with two coordinates.
        """
        if len(self.scaled[0]) != 2:
            raise ValueError("Cannot draw 3d graphic.\n"
                             "Reason: only two parameters must be optimized\n")

        x, y, z = self.__calculate_values(amount=1000, additional_shift=5)

        ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, **self.surface_config)

        colors = iter(cm.hot(np.flip(np.linspace(0, 1, len(self.scaled)))))
        for i in range(1, len(self.scaled)):
            color = next(colors)
            ax.plot([self.scaled[i - 1][0], self.scaled[i][0]],
                    [self.scaled[i - 1][1], self.scaled[i][1]],
                    [self.func(self.scaled[i - 1]), self.func(self.scaled[i])],
                    color=color, **self.line_config)

        # quadratic function, already calculated for all x and y
        levels = sorted(list(set([self.func(p) for p in self.scaled])))
        ax.contour(x, y, z, levels=levels, **self.contour_config)

        self.__set_headers(f'3d-converge - {self.method_name}')
        self.__complete_plot('3d_converge', show_image)

    def draw_2d(self, show_image=True):
        """
        Requires a function already built from x and y.
        Minimum point with two coordinates.
        """
        if len(self.scaled[0]) != 2:
            raise ValueError("Cannot draw 3d graphic projection.\n"
                             "Reason: only two parameters must be optimized\n")

        x, y, z = self.__calculate_values(amount=1000, additional_shift=5)

        levels = sorted(list(set([self.func(p) for p in self.scaled])))
        plt.contour(x, y, z, levels=levels, **self.contour_config)
        plt.plot(self.scaled[:, 0], self.scaled[:, 1], **self.line_config)

        plt.text(*self.scaled[0], self.__point_text(self.scaled[0]))
        plt.text(*self.scaled[-1], self.__point_text(self.scaled[-1]))

        self.__set_headers(f'2d-projection - {self.method_name}')
        self.__complete_plot('2d_projection', show_image)

    def draw_linear_regression(self, x, y, nth, shift=2, show_image=True):
        if len(x[0]) != 1 and len(y[0]) != 1:
            raise ValueError("Incorrect data provided")

        if len(self.rescaled[0]) != 2:
            raise ValueError("Two parameters must be optimized:\n"
                             "Line intercept and slope\n")

        scalars = self.rescaled[0::nth]

        self.__configure_plot(x, y, shift)

        plt.scatter(x, y, **self.scatter_config)

        x = np.linspace(np.amin(x) - shift, np.amax(x) + shift, 1000)

        color = iter(cm.coolwarm(np.linspace(0, 1, len(scalars))))
        for index, scalar in enumerate(scalars):
            y = scalar[0]
            for i in range(1, len(scalar)):
                y += x * scalar[i]
            plt.plot(x, y,
                     color=(next(color)),
                     linewidth=(3 if index == 0 or index == (len(scalars) - 1) else 0.6))

        self.__set_headers('Linear Regression')
        self.__complete_plot('linear_regression', show_image)

    def draw_2d_nonlinear_regression(self, xs, ys, shift=1, show_image=True):
        """
        Requires only the predicted height function.
        Possible many parameters.
        """
        first_scalar = self.rescaled[0]
        last_scalar = self.rescaled[-1]

        self.__configure_plot(xs, ys, shift)

        plt.scatter(xs, ys, **self.scatter_config)

        x = np.linspace(np.amin(xs) - shift, np.amax(xs) + shift, 1000)
        plt.plot(x, self.r(first_scalar, x), c='blue')
        plt.plot(x, self.r(last_scalar, x), c='red')

        self.__set_headers(f'2d Nonlinear Regression - {self.method_name}')
        self.__complete_plot('2d_nonlinear_regression', show_image)

    def draw_3d_nonlinear_regression(self, xs1, xs2, y, show_image=True):
        """
        Requires only the predicted height function.
        Possible many parameters.
        """
        last_scalar = self.rescaled[-1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x1, x2 = np.meshgrid(xs1, xs2)

        ax.scatter(x1, x2, y, **self.scatter_config)

        x = np.column_stack([x1.ravel(), x2.ravel()])
        ax.plot_surface(x1, x2, self.r(last_scalar, x).reshape(len(xs1), len(xs2)), **self.surface_config)

        self.__set_headers(f'3d Nonlinear Regression - {self.method_name}')
        self.__complete_plot('3d_nonlinear_regression', show_image)

    def __point_text(self, point):
        return f'({point[0]:.2f}, {point[1]:.2f}) - {self.func(point):.2f}'

    @staticmethod
    def __set_headers(title):
        ax = plt.gca()
        plt.title(title)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        if hasattr(ax, 'set_zlabel'):
            ax.set_zlabel('$Z$')

    @staticmethod
    def __configure_plot(xs, ys, shift):
        ax = plt.gca()
        ax.set_xlim([np.amin(xs) - shift, np.amax(xs) + shift])
        ax.set_ylim([np.amin(ys) - shift, np.amax(ys) + shift])

    def __complete_plot(self, directory, show_image):
        full_dir = f'{save_directory}/{directory}'
        os.makedirs(full_dir, exist_ok=True)
        plt.savefig(f'{full_dir}/scaler-{self.scaler_name}_method-{self.method_name}_batch-{self.batch_size}.png')

        if show_image:
            plt.show()

        plt.close()
