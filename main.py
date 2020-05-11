# INFO
# ranked data is treated same as numerical
# categorical data categories should be integers
# categories are all integers between (including) min and max that exist in file
# X is a list of columns
# datatypes should be set BEFORE loading dataset


import numpy as np
from enum import Enum
import itertools as itrt
import math


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


class MyAdaBoost:
    def __init__(self):
        self.X = []
        self.y = []
        self.weights = []
        self.datatypes = []
        self.stumps = []

    def load_dataset(self, filename):
        X = []
        y = []
        datafile = open(filename, encoding="utf-8")
        for line in datafile:
            rowvals = line.strip().split(",")[:-1]
            if len(X) == 0:
                for col in range(len(rowvals)):
                    X.append([])

            for i in range(len(X)):
                if self.datatypes[i] == DataType.BINARY:
                    if rowvals[i] == "0":
                        X[i].append(False)
                    else:
                        X[i].append(True)

                elif self.datatypes[i] == DataType.NUMERICAL:
                    X[i].append(float(rowvals[i]))
                elif self.datatypes[i] == DataType.CATEGORICAL:
                    X[i].append(int(rowvals[i]))

            yvalstr = line.strip().split(",")[-1]
            if yvalstr == "0":
                y.append(False)
            else:
                y.append(True)

        self.X = X
        self.y = y
        num = len(self.y)
        _num = 1 / num
        for i in range(num):
            self.weights.append(_num)

        datafile.close()

    def set_datatypes(self, datatype_string):
        temp = datatype_string.split(",")
        for char in temp:
            if char == "b":
                self.datatypes.append(DataType.BINARY)
            elif char == "n":
                self.datatypes.append(DataType.NUMERICAL)
            elif char == "c":
                self.datatypes.append(DataType.CATEGORICAL)

    def calc_terr(self, stump):
        total_error = 0
        for attr_val, label, weight in zip(self.X[stump.column], self.y, self.weights):
            decision = stump.decide(attr_val)
            if decision != label:
                total_error += weight
        return total_error

    def calc_aos(self, stump):
        total_error = self.calc_terr(stump)
        AoS = 0.5 * np.log((1 - total_error) / total_error)
        stump.AoS = AoS
        return AoS

    def calc_gini(self, stump):
        tp, fp, tn, fn = 0, 0, 0, 0
        for attr_val, label, weight in zip(self.X[stump.column], self.y, self.weights):
            decision = stump.decide(attr_val)
            if decision == label:
                if decision == 0:
                    tn += weight
                else:
                    tp += weight
            else:
                if decision == 0:
                    fn += weight
                else:
                    fp += weight

        possum = tp + fp
        negsum = tn + fn

        if possum == 0:
            gini_pos_leaf = 1
        else:
            gini_pos_leaf = 1 - (tp / (possum)) ** 2 - (fp / (possum)) ** 2

        if negsum == 0:
            gini_neg_leaf = 1
        else:
            gini_neg_leaf = 1 - (tn / (negsum)) ** 2 - (fn / (negsum)) ** 2

        gini = (possum * gini_pos_leaf + negsum * gini_neg_leaf) / (possum + negsum)
        return gini

    def pick_best_stump(self):
        # for every attribute column
        tempstumps = []
        tempginis = []
        for idx, datatype in enumerate(self.datatypes):
            tempstumps.append(DecisionStump(dtype=datatype, column=idx))
            curstump = tempstumps[-1]

            if datatype == DataType.NUMERICAL:
                temprange = []

                templist = self.X[idx][:]
                templist.sort()
                curstump.num_range = (templist[0], templist[-1])

                # try every threshold value
                thr_candidates = []
                for i in range(len(templist) - 1):
                    thr_candidates.append((templist[i] + templist[i + 1]) / 2)

                stumps_ginis = np.zeros((len(templist) - 1, 2))
                for i, thr in enumerate(thr_candidates):
                    curstump.threshold = thr
                    stumps_ginis[i, 0] = thr
                    stumps_ginis[i, 1] = self.calc_gini(curstump)

                # TODO improve performance
                stumps_ginis = stumps_ginis[np.argsort(stumps_ginis[:, 1])]

                # set best threshold value
                curstump.threshold = stumps_ginis[0, 0]
                tempginis.append(stumps_ginis[0, 1])

                # print(
                #     f"{idx}: NUMERICAL\tGINI = {tempginis[-1]}\tTHRESHOLD = {curstump.threshold}"
                # )

            elif datatype == DataType.CATEGORICAL:
                templist = range(min(self.X[idx]), max(self.X[idx]) + 1)

                combinations = []
                for i in range(1, len(templist)):
                    for combination in list(itrt.combinations(templist, i)):
                        combinations.append(combination)

                stumps_ginis = {}
                for i, combination in enumerate(combinations):
                    curstump.categories = combination
                    stumps_ginis[combination] = self.calc_gini(curstump)

                # TODO improve performance
                stumps_ginis = {
                    k: v
                    for k, v in sorted(stumps_ginis.items(), key=lambda item: item[1])
                }

                # set best threshold value
                curstump.categories = list(stumps_ginis.keys())[0]
                tempginis.append(list(stumps_ginis.values())[0])

                # print(
                #     f"{idx}: CATEGORICAL\tGINI = {tempginis[-1]}\tCATEGORIES = {curstump.categories}"
                # )

            elif datatype == DataType.BINARY:
                gini = self.calc_gini(curstump)
                tempginis.append(gini)
                # print(f"{idx}: BINARY\tGINI = {tempginis[-1]}")

        tempdict = dict(zip(tempstumps, tempginis))
        tempdict = {k: v for k, v in sorted(tempdict.items(), key=lambda item: item[1])}
        beststump = list(tempdict.keys())[0]
        bestgini = list(tempdict.values())[0]
        self.stumps.append(beststump)
        self.calc_aos(self.stumps[-1])
        print(
            f"best stump for column {self.stumps[-1].column} with gini {bestgini} and AoS {self.stumps[-1].AoS} of type {self.stumps[-1].dtype}"
        )

    def recalc_weights(self, stump):
        for idx, (attr_val, label) in enumerate(zip(self.X[stump.column], self.y)):
            if stump.decide(attr_val) == label:
                self.weights[idx] = self.weights[idx] * (math.e ** -stump.AoS)
            else:
                self.weights[idx] = self.weights[idx] * (math.e ** stump.AoS)

        weightsum = 0
        for weight in self.weights:
            weightsum += weight

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] / weightsum

    def generate_n_stumps(self, n):
        for i in range(n):
            self.pick_best_stump()
            self.recalc_weights(self.stumps[-1])

    def decide(self, input_sample):
        votecounter = 0
        for stump in self.stumps:
            decision = stump.decide(input_sample[stump.column])
            if decision:
                votecounter += stump.AoS
            else:
                votecounter -= stump.AoS

        if votecounter > 0:
            return True
        else:
            return False


for iteration in range(1, 11):
    print(f"# FOREST WITH {iteration} TREES #")
    myclf = MyAdaBoost()
    myclf.set_datatypes("b,n,n,c,c,c,n,b,b,n,b,c,n,n")
    myclf.load_dataset("australian.csv")
    myclf.generate_n_stumps(iteration)

    match_cntr = 0
    for i in range(len(myclf.y)):
        sample = []
        label = myclf.y[i]
        for col in myclf.X:
            sample.append(col[i])

        decision = myclf.decide(sample)
        # print(f"DECISION: {decision}\tLABEL: {label}")
        if decision == label:
            match_cntr += 1

    print(f"SUCCESS RATE: {round(match_cntr/len(myclf.y)*100,2)}%\n")
