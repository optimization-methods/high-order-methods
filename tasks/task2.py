def draw(descent_result, show=False):
    drawer = Drawer(descent_result)

    # noinspection SpellCheckingInspection
    drawer.draw_3d_pigis('pigis', show)
    drawer.pigis(show)


# noinspection SpellCheckingInspection
def test_bfgs_easy():
    draw_and_save(BfgsDescentMethod(1, 2, 3, 'pigis'))


# noinspection SpellCheckingInspection
def test_bfgs_medium():
    draw_and_save(BfgsDescentMethod(4, 5, 6, 'swingis'))


# noinspection SpellCheckingInspection
def test_bfgs_hard():
    draw_and_save(BfgsDescentMethod(7, 8, 9, 'pigletis'))


def compare_with_previous():
    pass

# test_bfgs_easy()
# test_bfgs_medium()
# test_bfgs_hard()
# compare_with_previous()
