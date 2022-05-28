import csv

from utils import config
from utils.dataset import Dataset


class DatasetReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def parse(self):
        _input = []
        with open(f'{config.source}\\datasets\\{self.file_name}\\input.csv', newline='') as file:
            for row in csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC):
                _input.append(row)

        _output = []
        with open(f'{config.source}\\datasets\\{self.file_name}\\output.csv', newline='') as file:
            for row in csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC):
                if len(row) != 1:
                    raise ValueError('Row length must be 1')

                _output.append(row[0])

        if len(_input) != len(_output):
            raise ValueError('X and U must have the same length')

        return Dataset(_input, _output)
