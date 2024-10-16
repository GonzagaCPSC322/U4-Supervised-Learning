import operator
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier

from mysimplelinearregressor import MySimpleLinearRegressor

# unit under test
def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

# unit test
def test_compute_euclidean_distance():
    # we need data
    np.random.seed(0)
    v1 = np.random.random(100)
    v2 = np.random.random(100)
    dist = compute_euclidean_distance(v1, v2)
    # let's test against scipy
    sp_dist = euclidean(v1, v2)
    assert np.isclose(dist, sp_dist)

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]
            
def main():
    # starting with PA4, we are going to implement
    # common ML algorithms adhering to a popular
    # (e.g., sci-kit learn, tensorflow, etc.) API
    # API: application programming interface
    # https://scikit-learn.org/1.3/tutorial/statistical_inference/supervised_learning.html

    # X: 2D feature matrix (e.g. rows are instances of
    # attribute values)
    # y: 1D class vector (e.g. the values you want to 
    # predict)
    # note: your class values (e.g. y) are stored
    # separately from your feature matrix (e.g. X)

    # we typically divide X and y into 
    # "training" and "testing" sets
    # we build model/algorithm using a training set
    # we evaluate a model/algorithm using a testing set
    # X_train and y_train are parallel
    # X_test and y_test are parallel

    # each algorithm is implemented as a class
    # the class has 2 common methods
    # fit(X_train, y_train) -> None
    # fit() "fits" a model/algorithm to the training set
    # predict(X_test) -> y_predicted(list)
    # predict() makes "predictions" for each instance
    # in the test set
    # y_predicted is parallel to y_test
    # use y_predicted and y_test to evaluate how
    # well the algorithm/model did on the test set
    # regression: example metric: MAE (mean abs error)
    # the average of the aboslute differences
    # classification: example metric: accuracy
    # is the number of correct predictions / total 
    # number of predictions

    # mysimplelinearregressor.py is our simple linear
    # regression code from last class, refactored
    # to follow this class design
    # let's see the API in action!!
    # we need X_train and y_train data
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    # print(X_train)
    # print(y_train)

    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train)
    X_test = [[150], [175]]
    y_test = [300, 350]
    y_pred = lin_reg.predict(X_test)
    print(y_pred)

    # lets convert this code to unit tests!

    # PA4 kNN starter code
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
    unseen_instance = [2, 3]
    # 1. normalize if needed (not needed for dataset, assumed [0, 10])
    # 2. calculate distances between each training set instance
    # and the unseen instance
    row_indexes_dists = [] # list of tuple pairs (row index, dist)
    for i, row in enumerate(X_train):
        dist = compute_euclidean_distance(row, unseen_instance)
        row_indexes_dists.append((i, dist))
    # need to sort row_indexes_dists by dist
    print(row_indexes_dists)
    row_indexes_dists.sort(key=operator.itemgetter(-1)) # get the item
    # in the tuple at index -1 and use that for sorting
    k = 3 
    top_k = row_indexes_dists[:k]
    for row in top_k:
        print(row)

    # TODO: extract the top k closes neighbors' y labels from y_train
    # then use majority voting to find the prediction for this test instance
    # can make use of get_frequencies()

    # check work against sci-kit learn
    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    distances, indexes = knn_clf.kneighbors([unseen_instance])
    print(distances)
    print(indexes)

    # PA5 starter code
    # Prompt: In ClassificationFun/main.py, write a function called `randomize_in_place(alist, parallel_list=None)`
    # that accepts at least one list and shuffles the elements of the list. 
    # If a second list is passed in, it should be shuffled in the same order.
    # Call your function with the `X_train` and `y_train` lists. Make sure they get shuffled in parallel :)
    # Note: this function will be super handy for PA5! I'll post a solution after class :)
    np.random.seed(0)
    print("randomizing in place")
    randomize_in_place(X_train, y_train)
    for i in range(len(X_train)):
        # check they are still parallel
        print(X_train[i], y_train[i])

if __name__ == "__main__":
    main()