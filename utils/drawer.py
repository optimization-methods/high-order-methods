import os

import numpy as np
from matplotlib import pyplot as plt, cm
import utils.config as config

save_directory = f"{config.source}/images"


class SurfaceConfig(object):
    def __init__(self, alpha, color):
        self.alpha = alpha
        self.color = color


# noinspection SpellCheckingInspection
class ContourConfig(object):
    def __init__(self, cmap, alpha, linestyles, linewidths):
        self.cmap = cmap
        self.alpha = alpha
        self.linestyles = linestyles
        self.linewidths = linewidths


# noinspection SpellCheckingInspection
class LineConfig(object):
    def __init__(self, cmap, alpha, marker, markersize, linestyle, linewidth):
        self.cmap = cmap
        self.alpha = alpha
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize


class Drawer(object):
    def __init__(self, sgd_result):
        self.scaled = np.array(sgd_result.scaled_scalars, dtype=config.dtype)
        self.rescaled = np.array(sgd_result.rescaled_scalars, dtype=config.dtype)
        self.func = sgd_result.func

        self.r = sgd_result.r

        self.batch_size = sgd_result.batch_size

        self.scaler_name = sgd_result.scaler_name
        self.method_name = sgd_result.method_name

        self.surface_config = SurfaceConfig(alpha=0.2, color='green')
        self.line_config = LineConfig(
            lambda lightness: cm.hot(lightness),
            alpha=1, marker='o', markersize=3, linestyle='solid', linewidth=2
        )
        self.contour_config = ContourConfig(cmap='autumn', alpha=0.8, linestyles='dashed', linewidths=0.5)

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

    '''
    Requires a function already built from x and y.
    Minimum point with two coordinates.
    '''
    def draw_3d(self, show_image=True):
        if len(self.scaled[0]) != 2:
            raise ValueError("Cannot draw 3d graphic.\n"
                             "Reason: only two parameters must be optimized\n")

        x, y, z = self.__calculate_values(amount=1000, additional_shift=5)

        ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z,
                        color=self.surface_config.color,
                        alpha=self.surface_config.alpha)

        colors = iter(self.line_config.cmap(np.flip(np.linspace(0, 1, len(self.scaled)))))
        for i in range(1, len(self.scaled)):
            color = next(colors)
            ax.plot([self.scaled[i - 1][0], self.scaled[i][0]],
                    [self.scaled[i - 1][1], self.scaled[i][1]],
                    [self.func(self.scaled[i - 1]), self.func(self.scaled[i])],
                    alpha=self.line_config.alpha,
                    color=color,
                    marker=self.line_config.marker,
                    markersize=self.line_config.markersize,
                    linestyle=self.line_config.linestyle,
                    linewidth=self.line_config.linewidth)

        # quadratic function, already calculated for all x and y
        levels = sorted(list(set([self.func(p) for p in self.scaled])))
        ax.contour(x, y, z, levels=levels,
                   alpha=self.contour_config.alpha,
                   cmap=self.contour_config.cmap,
                   linestyles=self.contour_config.linestyles,
                   linewidths=self.contour_config.linewidths)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')

        self.__complete_plot('3d', show_image)

    '''
    Requires a function already built from x and y.
    Minimum point with two coordinates.
    '''
    def draw_2d(self, show_image=True):
        if len(self.scaled[0]) != 2:
            raise ValueError("Cannot draw 3d graphic projection.\n"
                             "Reason: only two parameters must be optimized\n")

        x, y, z = self.__calculate_values(amount=1000, additional_shift=5)

        levels = sorted(list(set([self.func(p) for p in self.scaled])))
        plt.contour(x, y, z, levels=levels)
        plt.plot(self.scaled[:, 0], self.scaled[:, 1],
                 marker=self.line_config.marker,
                 linestyle=self.line_config.linestyle,
                 linewidth=self.line_config.linewidth)

        plt.text(*self.scaled[0], self.__point_text(self.scaled[0]))
        plt.text(*self.scaled[-1], self.__point_text(self.scaled[-1]))

        plt.title(f'2d-projection - {self.method_name}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        self.__complete_plot('./img/2d', show_image)

    def draw_linear_regression(self, x, y, nth, shift=2, show_image=True):
        if len(x[0]) != 1 and len(y[0]) != 1:
            raise ValueError("Incorrect data provided")

        if len(self.rescaled[0]) != 2:
            raise ValueError("Two parameters must be optmized:\n"
                             "Line intercept and slope\n")

        scalars = self.rescaled[0::nth]

        ax = plt.gca()
        ax.set_xlim([np.amin(x) - shift, np.amax(x) + shift])
        ax.set_ylim([np.amin(y) - shift, np.amax(y) + shift])

        plt.scatter(x, y, color='green')

        color = iter(cm.coolwarm(np.linspace(0, 1, len(scalars))))
        for index, scalar in enumerate(scalars):
            c = next(color)
            x = np.linspace(np.amin(x) - shift, np.amax(x) + shift, 1000)
            y = scalar[0]
            for i in range(1, len(scalar)):
                y += x * scalar[i]

            plt.plot(
                x,
                y,
                color=c,
                linewidth=(3 if index == 0 or index == (len(scalars) - 1) else 0.6)
            )

        plt.title('Linear Regression')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        self.__complete_plot('linear_regression', show_image)

    '''
    Requires only predicted height function.
    Possible many parameters.
    '''
    def draw_2d_nonlinear_regression(self, xs, ys, shift=1, show_image=True):
        first_scalar = self.rescaled[0]
        last_scalar = self.rescaled[-1]

        ax = plt.gca()
        ax.set_xlim([np.amin(xs) - shift, np.amax(xs) + shift])
        ax.set_ylim([np.amin(ys) - shift, np.amax(ys) + shift])

        plt.scatter(xs, ys, color='green')
        X = np.linspace(np.amin(xs) - shift, np.amax(xs) + shift, 1000)
        # predicted height function
        plt.plot(X, self.r(first_scalar, X), c='blue')
        plt.plot(X, self.r(last_scalar, X), c='red')

        plt.title(f'2d_Nonlinear Regression - {self.method_name}')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        self.__complete_plot('2d_nonlinear_regression', show_image)

    '''
    Requires only predicted height function.
    Possible many parameters.
    '''
    def draw_3d_nonlinear_regression(self, X1, X2, Y, show_image=True):
        last_scalar = self.rescaled[-1]
        print(last_scalar)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X1, X2, Y, c='green')
        x = np.column_stack([X1.ravel(), X2.ravel()])
        ax.plot_surface(X1, X2, self.r(last_scalar, x).reshape(len(X1), len(X2)), cmap=cm.coolwarm)

        ax.text2D(0.05, 0.95, f'3d Nonlinear Regression - {self.method_name}', transform=ax.transAxes)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')

        self.__complete_plot('3d_nonlinear_regression', show_image)

    def __point_text(self, point):
        return f'({point[0]:.2f}, {point[1]:.2f}) - {self.func(point):.5f}'

    def __complete_plot(self, directory, show_image):
        full_dir = f'{save_directory}/{directory}'
        os.makedirs(full_dir, exist_ok=True)
        plt.savefig(f'{full_dir}/scaler-{self.scaler_name}_method-{self.method_name}_batch-{self.batch_size}.png')

        if show_image:
            plt.show()

        plt.close()
