import numpy as np
from enum import Enum

class DataType(Enum):
    BINARY = 1
    NUMERICAL = 2   #ranked data is treated same as numerical
    CATEGORICAL = 3

class DecisionStump():
    def __init__(self, dtype = DataType.NUMERICAL, threshold = 0, categories = []):
        self.dtype = dtype
        self.threshold = threshold
        self.categories = categories

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
        

class MyAdaBoost():
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.datatypes = []
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

    #TODO - add weights
    def calc_gini(self, stump, invals, outvals):
            tp, fp, tn, fn = 0,0,0,0
            for attr_val, label in zip(invals, outvals):                
                if label == '0':
                    label = False
                else:
                    label = True
                if stump.dtype == DataType.BINARY:
                    if attr_val == '0':
                        attr_val = False
                    else:
                        attr_val = True
                if stump.dtype == DataType.NUMERICAL:
                    attr_val = float(attr_val)

                if stump.dtype == DataType.CATEGORICAL:
                    attr_val = int(attr_val)

                decision = stump.decide(attr_val)

                if decision == label:
                    if decision == 0:
                        tn += 1
                    else:
                        tp += 1
                else:
                    if decision == 0:
                        fn += 1
                    else:
                        fp += 1

            possum = tp + fp
            negsum = tn + fn
            
            if possum == 0:
                gini_pos_leaf = 1
            else:
                gini_pos_leaf = 1 - (tp/(possum))**2 - (fp/(possum))**2
            
            if negsum == 0:
                gini_neg_leaf = 1
            else:
                gini_neg_leaf = 1 - (tn/(negsum))**2 - (fn/(negsum))**2
            
            gini = ( possum*gini_pos_leaf + negsum*gini_neg_leaf ) / (possum+negsum)
            return gini

    

    def pick_a_stump(self):
        #for every attribute column
        tempstumps = []
        tempginis = []
        for idx, datatype in enumerate(self.datatypes):
            tempstumps.append( DecisionStump(dtype=datatype) )
            curstump = tempstumps[-1]

            if curstump.dtype == DataType.NUMERICAL:
                templist = []
                for val in self.X[:,idx]:
                    templist.append(float(val))
                templist.sort()

                #try every threshold value
                thr_candidates = []
                for i in range(len(templist)-1):
                    thr_candidates.append((templist[i]+templist[i+1])/2)

                stumps_ginis = np.zeros((len(templist)-1, 2))
                for i, thr in enumerate(thr_candidates):
                    curstump.threshold = thr
                    stumps_ginis[i,0] = thr
                    stumps_ginis[i,1] = self.calc_gini(curstump,invals=self.X[:,idx], outvals=self.y)

                stumps_ginis = stumps_ginis[np.argsort(stumps_ginis[:, 1])] #TODO improve performance
                
                #set best threshold value
                curstump.threshold = stumps_ginis[0,0]
                tempginis.append(stumps_ginis[0,0])




            elif datatype == DataType.CATEGORICAL:
                print('not implemented yet, exitting....')
                continue

            elif datatype == DataType.BINARY:
                gini = self.calc_gini(stump=curstump,invals=self.X[:,idx], outvals=self.y)
                tempginis.append(gini)


            #print(f'{idx} > gini:{round(gini,4)}')
            #print(f'gini_pos:{round(gini_pos_leaf,2)}\tgini_neg:{round(gini_neg_leaf,2)}\tgini:{round(gini,2)}')

myclf = MyAdaBoost()
myclf.load_dataset('australian.csv')
myclf.set_datatypes('c,n,n,c,c,c,n,c,c,n,c,c,n,n')
myclf.pick_a_stump()


#create every tree

#choose the/another one with best gini index <--
#calculate amount of say                       |
#recalculate weights ---------------------------


# ds = DecisionStump()
# ds.set_datatype(DataType.CATEGORICAL)
# ds.set_categories([3,5,6,8])
# print(ds.decide(9))