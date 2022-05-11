# noinspection SpellCheckingInspection
class LBfgsDescentMethod(DescentMethod):
    def __init__(self, config):
        config.fistingate()

    def converge(self):
        return DescentResult('pigis')