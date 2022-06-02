class Data:
    def __init__(self, f, xs, ys, start):
        self.f = f
        self.xs = xs
        self.ys = ys
        self.start = start

class ThreeDimensionalData(Data):
    def __init__(self, f, xs, ys, start, x1, x2):
        super().__init__(f, xs, ys, start)
        self.x1 = x1
        self.x2 = x2
