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
        self.X = np.array([])
        self.datatypes = np.array([])
        self.y = np.array([])
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

    def set_datatypes(self, datatype_string):
        temp = datatype_string.split(',')
        if len(temp) != len(self.X[0]):
            print('error: incorrect number of datatypes')
            return

        self.datatypes = []

        for char in temp:
            if char == 'b':
                self.datatypes.append(DataType.BINARY)
            elif char == 'n':
                self.datatypes.append(DataType.NUMERICAL)
            elif char == 'c':
                self.datatypes.append(DataType.CATEGORICAL)

        self.datatypes = np.array(self.datatypes)

         

myclf = MyAdaBoost()
myclf.load_dataset('australian.csv')
myclf.set_datatypes('c,n,n,c,c,c,n,c,c,n,c,c,n,n')

#create every tree

#choose the/another one with best gini index <--
#calculate amount of say                       |
#recalculate weights ---------------------------


# ds = DecisionStump()
# ds.set_datatype(DataType.CATEGORICAL)
# ds.set_categories([3,5,6,8])
# print(ds.decide(9))