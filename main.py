import numpy as np
from enum import Enum

class DataType(Enum):
    BINARY = 1
    NUMERICAL = 2   #ranked data is treated same as numerical
    CATEGORICAL = 3

class DecisionStump():
    def __init__(self):
        self.type = DataType.NUMERICAL
        self.threshhold = 0

    def set_datatype(self, type):
        self.type = type

    def set_threshold(self, threshhold):
        self.threshhold = threshhold

    #list of all True categories
    def set_categories(self, categories):
        if type(categories) != np.ndarray:
            categories = np.array(categories)
        self.categories = categories

    def decide(self, input_val):
        if self.type == DataType.BINARY:
            if input_val:
                return True
            return False

        if self.type == DataType.NUMERICAL:
            if input_val > self.threshhold:
                return True
            return False

        if self.type == DataType.CATEGORICAL:
            if input_val in self.categories:
                return True
            return False
        


class MyAdaBoost():
    def __init__(self):
        self.X = []
        self.y = []
        self.stumps = []

    def load_dataset(self, filename):
        X = []
        y = []
        datafile = open(filename, encoding='utf-8')
        for line in datafile:
            X.append( line.strip().split(',')[:-1] )
            y.append( line.strip().split(',')[-1] )
        self.X = np.array(X)
        self.y = np.array(y)
         

# myclf = MyAdaBoost()
# myclf.load_dataset('australian.csv')
# #myclf.fit()

# ds = DecisionStump()
# ds.set_datatype(DataType.CATEGORICAL)
# ds.set_categories([3,5,6,8])
# print(ds.decide(9))