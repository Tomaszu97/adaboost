import numpy as np
import itertools as itrt
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .mystump import *


# scikit-learn adaboost: decisiontree
# AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
class MyAdaBoost(BaseEstimator, ClassifierMixin):
    """
    Our implementation of AdaBoost
    """

    def __init__(self, n_estimators=5, verbose=False):
        self.n_estimators = n_estimators
        self.X = []
        self.y = []
        self.weights = []
        self.datatypes = []
        self.stumps = []
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose:
            print("[MyAdaBoost] info: training started")
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if self.datatypes == []:
            print("[MyAdaBoost] error: set datatypes first!")
            return

        # convert to desired storage format
        self.X = []
        self.y = []
        for row in X:
            if len(self.X) == 0:
                for col in range(len(row)):
                    self.X.append([])

            for i in range(len(self.X)):
                if self.datatypes[i] == DataType.BINARY:
                    self.X[i].append(bool(row[i]))
                elif self.datatypes[i] == DataType.NUMERICAL:
                    self.X[i].append(float(row[i]))
                elif self.datatypes[i] == DataType.CATEGORICAL:
                    self.X[i].append(int(row[i]))

        for val in y:
            self.y.append(bool(val))

        # calculate initial weights
        num = len(self.y)
        _num = 1 / num
        for i in range(num):
            self.weights.append(_num)

        # training

        self.generate_n_stumps(self.n_estimators)

        if self.verbose:
            print("[MyAdaBoost] info: training finished")

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        y_pred = []
        for row in X:
            y_pred.append(self.decide(row))

        return np.array(y_pred)

    def load_dataset(self, filename):
        if self.datatypes == []:
            print("[myAdaBoost]error: Set datatypes first!")
            return

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
        if self.verbose:
            print(
                f"[MyAdaBoost] info: best stump for column {self.stumps[-1].column} with gini {round(bestgini,3)} and AoS {round(self.stumps[-1].AoS,3)} of type {self.stumps[-1].dtype}"
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
