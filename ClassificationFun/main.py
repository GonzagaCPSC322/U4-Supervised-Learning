import operator
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier

from mysimplelinearregressor import MySimpleLinearRegressor

def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def test_compute_euclidean_distance():
    # we need some data
    np.random.seed(0)
    v1 = list(np.random.random(100))
    v2 = list(np.random.random(100))

    dist = compute_euclidean_distance(v1, v2)
    # test against scipy
    dist_sol = euclidean(v1, v2)
    assert np.isclose(dist, dist_sol)

def main():
    # starting with PA4, we are going to be implementing
    # ML algorithms using a standard API
    # (application programming interface)
    # modeled after common ML libraries (sci-kit learn,
    # tensorflow, ...)
    # data convention
    # X: feature matrix (2D list AKA table where the columns
    # are features AKA attributes)
    # the class attribute is stored separatedly in y
    # y: target vector or class vector (1D list of the values
    # you want to predict)
    # X and y are parallel
    # we build a machine learning algorithm/model using "training data"
    # we evaluate a machine learning algorith/model using "testing data"
    # split X and y
    # X_train and y_train (are parallel)
    # X_test and y_test (are parallel)

    # we will implement algorithms as classes
    # algorithm convention
    # each class will implement a common public API
    # with two methods
    # fit(X_train, y_train) -> None
    # "fits" the model (AKA trains the algorithm)
    # using training data
    # predict(X_test) -> y_predicted(list)
    # makes "predictions" for unseen instances in the training data
    # y_predicted and y_test are parallel
    # we can determine how "good" the model algorithm is
    # by comparing these two lists
    # regression: MAE (mean absolute error)
    # average of the absolute values of the differences
    # between each pair in y_predicted and y_test
    # classification: accuracy
    # # of matching pairs in y_predicted and y_test / # of instances
    # in the test set

    # task: go to ClassificationFun on Github and grab
    # mysimplelinearregressor.py file

    # lets see an example of using the API
    # we need training data
    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    # print(X_train)
    # print(y_train)
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train) # "fits" slope (m) and intercept (b)
    # to the training data
    # lets get some predictions
    X_test = [[150], [200]]
    y_predicted = lin_reg.predict(X_test)
    print(y_predicted)

    # now, lets rework this example into unit testing form
    
    # kNN starter code and hints
    header = ["att1", "att2"]
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    test_instance = [2, 3]

    # calculate the distances between each train instance and the test
    # instance
    row_indexes_dists = []
    for i, train_instance in enumerate(X_train):
        dist = compute_euclidean_distance(train_instance, test_instance)
        row_indexes_dists.append((i, dist))

    for row in row_indexes_dists:
        print(row)
    # now we need the k smallest distances
    # we can sort row_indexes_dists by distance
    row_indexes_dists.sort(key=operator.itemgetter(-1)) # -1 or 1
    # because the distance is at the index in each item (list)
    
    # now, grab the top k
    k = 3
    top_k = row_indexes_dists[:k]
    print("top k:")
    for row in top_k:
        print(row)
    # TODO: extract the top k closes neighbors' y labels from y_train
    # then use majority voting to find the prediction for this test instance
    # can make use of get_frequencies()
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([test_instance])
    print(distances)
    print(indexes)



    # notes on generating training and testing sets from a dataset
    # 1. holdout method
    # 2. random subsampling
    # 3. k fold cross validation (and variants)
    # 4. bootstrap method

    # 1. holdout method
    # "hold out" a certain number of instances for testing
    # train on the remaining instances (e.g. the ones not held out)
    # test_size is the parameter that dictates how many instances
    # to hold out
    # e.g. test_size=2 -> holdout 2 instances for testing
    # e.g. test_size=0.33 -> holdout 33% of instances for testing
    # test_size=0.33 is pretty common
    # 2:1 train:test split

    # 2. random subsampling
    # repeat the holdout method k times
    # (this is a diff k than the k in kNN)
    # the accuracy is the average accuracy over
    # the k holdout methods

    # 3. k fold cross validation
    # (also a diff k...)
    # we are more intentional about our partitions
    # of the data into training and testing sets
    # each instance is in the test set once
    # create k folds
    # for each fold in the folds:
    #     hold out the fold for testing
    #     training on the remaining folds (folds - fold)
    # accuracy is the total correctly predicted divided
    # by the total predicted over all the folds
    # variants
    # LOOCV leave one out cross validation
    # k = N (number of instances in the dataset)
    # when you have a small dataset and you need as much
    # training data as you can get
    # stratified k fold cross validation
    # where each fold has roughly the same distribution of 
    # class labels as the original dataset
    # first, group by class
    # then for each group distribute the instances to the folds

    # 4. bootstrap method
    # like random subsampling but with replacement
    # create a training set by sampling N instances
    # with replacement 
    # N is the number instances in the dataset
    # the instances not sampled form your test set
    # ~63.2% of instances will be sampled into training set
    # ~36.8% of instances will not (form test set)
    # see github for math intuition
    # repeat the bootstrap sampling k times
    # accuracy is the weighted average accuracy
    # over the k runs
    # (weighted because test set size varies over k runs)



if __name__ == "__main__":
    main()