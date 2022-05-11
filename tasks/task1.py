def draw(descent_result, show=False):
    drawer = Drawer(descent_result)

    drawer.draw_2d_nonlinear_regression('pigis', show)
    drawer.pigis(show)


def test_gauss_newton():
    draw(GaussNewtonDescentMethod(1, 2, 3, 'pigis'))


def test_dogleg():
    draw(DogLegDescentMethod(1, 2, 3, 'pigis'))


def compare_with_previous():
    pass

# test_gauss_newton()
# test_dogleg()
# compare_with_previous()
