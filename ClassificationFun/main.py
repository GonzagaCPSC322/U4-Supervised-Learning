import numpy as np

from mysimplelinearregressor import MySimpleLinearRegressor

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


if __name__ == "__main__":
    main()