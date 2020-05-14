from enum import Enum


class DataType(Enum):
    BINARY = 1
    NUMERICAL = 2
    CATEGORICAL = 3


class DecisionStump:
    def __init__(
        self,
        dtype=DataType.NUMERICAL,
        threshold=0,
        num_range=(0, 100),
        categories=(),
        column=None,
    ):
        self.dtype = dtype
        self.threshold = threshold
        self.categories = categories
        self.range = num_range
        self.AoS = None
        self.column = column

    def decide(self, input_val):
        if self.dtype == DataType.BINARY:
            if input_val:
                return True
            return False

        if self.dtype == DataType.NUMERICAL:
            if input_val > self.threshold:
                return True
            return False

        if self.dtype == DataType.CATEGORICAL:
            if input_val in self.categories:
                return True
            return False
