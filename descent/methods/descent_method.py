from descent.methods.descent_result import DescentResult


class DescentMethod:
    def converge(self) -> DescentResult:
        raise NotImplementedError("You need to inherit this class")

    @property
    def name(self):
        raise NotImplementedError("You need to inherit this class")

    @property
    def xs(self):
        raise NotImplementedError("You need to inherit this class")

    @property
    def ys(self):
        raise NotImplementedError("You need to inherit this class")
