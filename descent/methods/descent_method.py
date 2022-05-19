from descent.methods.descent_result import DescentResult


class DescentMethod:
    def converge(self) -> DescentResult:
        raise NotImplementedError("You need to inherit this class")
