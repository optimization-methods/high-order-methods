import csv


class DatasetReader(object):
    def __init__(self, file_name):
        self._input = []
        with open(f'..\\..\\datasets\\{file_name}\\input.csv', newline='') as file:
            for row in csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC):
                self._input.append(row)

        self._output = []
        with open(f'..\\..\\datasets\\{file_name}\\output.csv', newline='') as file:
            for row in csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC):
                if len(row) != 1:
                    raise ValueError('Row length must be 1')

                self._output.append(row[0])

        if len(self._input) != len(self._output):
            raise ValueError('X and U must have the same length')

    @property
    def data(self):
        return self._input, self._output

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output
