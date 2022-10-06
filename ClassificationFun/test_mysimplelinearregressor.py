import numpy as np
from sklearn.linear_model import LinearRegression

from mysimplelinearregressor import MySimpleLinearRegressor

# test modules and test functions (AKA unit tests)
# with pytest start with test_

def test_mysimplelinearregressor_fit():
    # use assert statements to form test cases
    # start with simple/common test cases
    # then move on to complex/edge test cases

    # let's write test cases for fit()
    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train) # "fits" slope (m) and intercept (b)

    # 2 main ways to write asserts
    # 1. assert against "desk calculation" or "desk check"
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    # order: actual, expected (solution)
    assert np.isclose(lin_reg.slope, slope_solution)
    assert np.isclose(lin_reg.intercept, intercept_solution)

    # 2. assert against a known correct implementation
    # let's assert against slope and intercept values
    # from sci-kit learn's LinearRegression
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    assert np.isclose(lin_reg.slope, sklearn_lin_reg.coef_[0]) 
    #[0] because only 1 coefficient because only 1 feature
    assert np.isclose(lin_reg.intercept, sklearn_lin_reg.intercept_)

    # TODO: should probably add more test cases...

def test_mysimplelinearregressor_predict():
    # task: try to write the body of this unit test
    # assert against sklearn's predictions

    np.random.seed(0)
    X_train = [[value] for value in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train) # "fits" slope (m) and intercept (b)
    X_test = [[150], [200]]
    y_predicted = lin_reg.predict(X_test)

    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)
    assert np.allclose(y_predicted, sklearn_y_predicted)
