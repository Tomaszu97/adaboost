#
#!!!!!!!!!!!!!!! fix mock types in australian !!!!!!!!!!

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from sklearn import datasets, model_selection, naive_bayes, metrics
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from mylib.myadaboost import *

# CONFIG
n_stumps = 3
n_folds = 5
my_ada_verbose = False
datasets = ["australian", "banknote"]
# /CONFIG


print("# MyAdaBoost vs Scikit's AdaBoost - comparison")

for dataset_name in datasets:
    print(f"\n---\n## Dataset '{dataset_name}' ")

    # create new classifiers
    clfs = {
        "ScikitAdaBoost": AdaBoostClassifier(n_estimators=n_stumps, random_state=123),
        "MyAdaBoost": MyAdaBoost(n_estimators=n_stumps, verbose=my_ada_verbose),
    }

    # load current dataset from file
    dataset = f"./datasets/{dataset_name}.csv"
    dataset = np.genfromtxt(dataset, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # test both classifiers for every fold
    folds = n_folds
    results = np.zeros((len(clfs), folds))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1410)
    for fold, (train, test) in enumerate(skf.split(X, y)):
        for clf_idx, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])

            # set datatypes for our implementation
            if clf_name == "MyAdaBoost":
                clf.set_datatypes(
                    open(f"./datasets-types/{dataset_name}.types", encoding="utf-8")
                    .readline()
                    .strip()
                )

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            results[clf_idx, fold] = accuracy_score(y[test], y_pred)

    # print results
    for clf_idx, clf_name in enumerate(clfs):
        results_vector = results[
            clf_idx,
        ]

        rounded_results = [round(num, 3) for num in results_vector]

        print(f"### {clf_name}")
        print(f"accuracy results : {rounded_results}")
        print(f"accuracy (average): {round(np.average(results_vector), 3)}")
        print(f"accuracy (std deviation): {round(np.std(results_vector), 3)}")

    print("### Statistical comparison")

    if np.array_equal(results[0], results[1]):
        print(
            "No need to check - results are identical. Classifiers performed equally."
        )
    else:
        alpha = 0.05
        test = ttest_rel(results[0], results[1])
        T = test.statistic
        p = test.pvalue

        print(f"alpha = {alpha}\nT = {T}\np={p}")

        if p > alpha:
            print(
                "No statistically important difference between classifiers (p > alpha)."
            )
        elif T > 0:
            print(f"{clfs.keys()[0]} is better (T > 0).")
        else:
            print(f"{clfs.keys()[1]} is better (p <= alpha & T <= 0).")
