def draw(descent_result, show=False):
    drawer = Drawer(descent_result)

    drawer.draw_real_pig_swini_porosenok_point('pigis', show)
    drawer.pigis(show)


def test_l_bfgs_easy():
    draw_and_save(LBbfsDescentMethod(1, 2, 3, 'pigis'))


def test_l_bfgs_medium():
    draw_and_save(LBfgsDescentMethod(4, 5, 6, 'swingis'))


def test_l_bfgs_hard():
    draw_and_save(LBfgsDescentMethod(7, 8, 9, 'pigletis'))


def compare_with_bfgs():
    pass

# test_l_bfgs_easy()
# test_l_bfgs_medium()
# test_l_bfgs_hard()
# compare_with_bfgs()
