from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from sklearn import datasets, model_selection, naive_bayes, metrics
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mylib.myadaboost import *
from sklearn.tree import export_text

### INFO ###
# THIS IMPLEMENTATION IS EXPERIMENTAL
# it is very slow to fit MyAdaBoost
# but it takes categorical data into account
# datasets are in datasets directory
# datatypes are specified in dataset-types directory (same file name as dataset, extension .type)

### HOWTO ###
# edit below config section
# run script
# some output is printed and also a output.md file is created


# CONFIG
n_stumps = 3
n_folds = 2
my_ada_verbose = True
datasets = ["wdbc"]
# /CONFIG


logfile = open("output.md", encoding="utf-8", mode="w+")


def log(line):
    print(line)
    logfile.write(line + "\n")


def skl_clf_print_stumps(clf):
    def skl_decisiontree_feature(dectree):
        treestr = export_text(dectree, spacing=1)
        return int(treestr.split(" ")[1].split("_")[1])

    def skl_decisiontree_threshold(dectree):
        treestr = export_text(dectree, spacing=1, decimals=3)
        return float(treestr.replace("\n", "").replace("|", "").split(" ")[3])

    counter = 0
    for i in range(clf.n_estimators):
        log(f"\tStump no. {counter}:")
        featurenum = skl_decisiontree_feature(clf.estimators_[i])
        AoS = clf.feature_importances_[featurenum]
        log(f"\t\tfeature:\t{featurenum}")
        log(f"\t\ttype:\t\tnumerical")
        log(
            f"\t\tthreshold:\t{round(skl_decisiontree_threshold(clf.estimators_[i]), 3)}"
        )
        log(f"\t\tAmount of Say:\t{round(AoS,3)}")
        counter += 1


def my_clf_print_stumps(clf):
    counter = 0
    for stump in clf.stumps:
        log(f"\tStump no. {counter}:")

        log(f"\t\tfeature:\t{stump.column}")

        if stump.dtype == DataType.BINARY:
            log(f"\t\ttype:\t\tbinary")
        elif stump.dtype == DataType.NUMERICAL:
            log(f"\t\ttype:\t\tnumerical")
            log(f"\t\tthreshold:\t{round(stump.threshold, 3)}")
        elif stump.dtype == DataType.CATEGORICAL:
            log(f"\t\ttype:\t\tcategorical")
            log(f"\t\tpool:\t{stump.categories}")

        log(f"\t\tAmount of Say:\t{round(stump.AoS,3)}")

        counter += 1


log("# MyAdaBoost vs Scikit's AdaBoost - comparison")
for dataset_name in datasets:
    log(f"\n---\n## Dataset '{dataset_name}' ")

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
        log(f"### Fold no. {fold}")
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

            log(f"**Classifier: {clf_name}**\n")

            if clf_name == "ScikitAdaBoost":
                skl_clf_print_stumps(clf)

            else:
                my_clf_print_stumps(clf)

    # print summary
    for clf_idx, clf_name in enumerate(clfs):
        results_vector = results[
            clf_idx,
        ]

        log(f"### {clf_name}")
        rounded_results = [round(num, 3) for num in results_vector]
        log(f"accuracy results : {rounded_results}")
        log(f"accuracy (average): {round(np.average(results_vector), 3)}")
        log(f"accuracy (std deviation): {round(np.std(results_vector), 3)}")

    log("### Statistical comparison")

    if np.array_equal(results[0], results[1]):
        log("No need to check - results are identical. Classifiers performed equally.")
    else:
        alpha = 0.05
        test = ttest_rel(results[0], results[1])
        T = test.statistic
        p = test.pvalue

        log(f"alpha = {alpha}\nT = {T}\np={p}")

        if p > alpha:
            log(
                "No statistically important difference between classifiers (p > alpha)."
            )
        elif T > 0:
            log(f"First classifier is better (T > 0).")
        else:
            log(f"Second classifier is better (p <= alpha & T <= 0).")


logfile.close()
