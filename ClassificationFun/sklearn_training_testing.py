import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix

# recreating the magnet demo we did in class with Sci-kit Learn functions/methods
# so you can see the input and output formats of these functions/methods,
# which will help you implement the functionality to take input of the same form
# and produce output of the same form
# note that the provided unit tests in test_myevaluation.py test against these
# Sci-kit Learn functions/methods so your implementation needs to match Sci-kit
# Learn's format

# little utility function
def display_train_test(label, X_train, X_test, y_train, y_test):
    print("***", label, "***")
    print("train:")
    for i, _ in enumerate(X_train):
        print(X_train[i], "->", y_train[i])
    print("test:")
    for i, _ in enumerate(X_test):
        print(X_test[i], "->", y_test[i])
    print()

# little utility function
def display_folds(label, folds, X, y):
    print(label)
    for i, _ in enumerate(folds):
        curr_fold = folds[i]
        print("fold #:", i)
        train_indexes = list(curr_fold[0])
        test_indexes = list(curr_fold[1])
        train_instances = [str(X[index]) + " -> " + str(y[index]) for index in train_indexes]
        test_instances = [str(X[index]) + " -> " + str(y[index]) for index in test_indexes]
        print("train indexes:", train_indexes)
        for instance_str in train_instances:
            print("\t" + instance_str)
        print("test indexes:", test_indexes)
        for instance_str in test_instances:
            print("\t" + instance_str)
        print()

# we aren't actually going to use this data for classification,
# just for tracing algorithms that split a dataset into training
# and testing, so I'll make dummy X data where the values of each
# instance make it really clear what its original row index in X
# was (so we can uniquely identify it and watch it move)
# and what it's associated y label is (green or yellow) (so we can
# make sure our Xs and ys stay parallel throughout)
X = [[0, "g"], [1, "g"], [2, "g"], [3, "g"], [4, "g"], [5, "y"], [6, "y"], [7, "y"], [8, "y"], [9, "y"]]
# green (pos)/yellow (neg) y labels
y = ["🟢", "🟢", "🟢", "🟢", "🟢", "🟡", "🟡", "🟡", "🟡", "🟡"]

# 1) HOLD OUT METHOD w/various parameters
# returns lists of instances
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, shuffle=False)
display_train_test("train_test_split w/test_size=2, shuffle=False", X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=0, shuffle=True)
display_train_test("train_test_split w/test_size=2, random_state=0, shuffle=True", X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle=True)
display_train_test("train_test_split w/test_size=0.33, random_state=0, shuffle=True", X_train, X_test, y_train, y_test)

# random subsampling
# TODO: repeat the hold out method k times w/different random_state

# K FOLD CROSS VALIDATION w/various parameters
# returns lists of indexes
n_splits = 5
standard_kf = KFold(n_splits=n_splits)
folds = list(standard_kf.split(X, y))
display_folds("k fold w/n_splits=5, shuffle=False", folds, X, y)

# note: # random_state=0 did not give folds with homogeneous colors in test, so using 1 because it
# motivates stratification
standard_kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
folds = list(standard_kf.split(X, y))
display_folds("k fold w/n_splits=5, shuffle=True", folds, X, y)

n_splits = 4
standard_kf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
folds = list(standard_kf.split(X, y))
display_folds("k fold w/n_splits=4, shuffle=True", folds, X, y)

# STRATIFIED K FOLD CROSS VALIDATION w/various parameters
# returns lists of indexes
n_splits = 5
stratified_kf = StratifiedKFold(n_splits=n_splits)
folds = list(stratified_kf.split(X, y))
display_folds("stratified k fold w/n_splits=5, shuffle=False", folds, X, y)

stratified_kf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
folds = list(stratified_kf.split(X, y))
display_folds("stratified k fold w/n_splits=5, shuffle=True", folds, X, y)

n_splits = 4
stratified_kf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
folds = list(stratified_kf.split(X, y))
display_folds("stratified k fold w/n_splits=4, shuffle=True", folds, X, y)

# BOOTSTRAP METHOD
# returns lists of instances
# repeat bootstrap sample k times w/different random_state
# (start of) bootstrap sample
X_train, y_train = resample(X, y, n_samples=len(X), random_state=0)
print("***", "bootstrap sample w/n_samples=N", "***")
print("train:")
for i, _ in enumerate(X_train):
    print(X_train[i], "->", y_train[i])
# TODO: X_test and y_test would correspond to the samples not included in X_train
# e.g. w/random_state=0, the test set would be the instances:
# [1, 'g'] -> 🟢, [6, 'y'] -> 🟡, [8, 'y'] -> 🟡

# ACCURACY SCORE
# suppose y is y_test (AKA y_true, y_actual):
y_test = y
# and the made up class labels below are y_predicted:
y_predicted = y.copy() # copy because shuffle does in place shuffle
random.seed(0)
random.shuffle(y_predicted)
print("y_test:\t\t", y)
print("y_predicted:\t", y_predicted)
accuracy = accuracy_score(y_test, y_predicted, normalize=True)
print("accuracy score w/normalize=True:", accuracy)
accuracy = accuracy_score(y_test, y_predicted, normalize=False)
print("accuracy score w/normalize=True:", accuracy)

# CONFUSION MATRIX
# labels is list of labels to index the matrix
matrix = confusion_matrix(y_test, y_predicted, labels=["🟢", "🟡"])
print("confusion matrix:\n", matrix)