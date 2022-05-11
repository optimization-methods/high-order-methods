class Dataset:
    def __init__(self, _input, _output):
        self._input = _input
        self._output = _output

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output
