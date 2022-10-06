import numpy as np

from mysimplelinearregressor import MySimpleLinearRegressor

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
    


if __name__ == "__main__":
    main()