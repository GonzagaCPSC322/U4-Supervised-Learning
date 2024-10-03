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
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train) # "fits" slope (m) and intercept (b)
     
    # 2 main ways to write asserts
    # 1. assert against "desk calculation" or "desk check"
    # order: actual, expected (solution)
    assert np.isclose(lin_reg.slope, 1.9249174584304438)
    assert np.isclose(lin_reg.intercept, 5.211786196055158)

    # 2. assert against a known correct implementation
    # let's assert against slope and intercept values
    # from sci-kit learn's LinearRegression
    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    #[0] because only 1 coefficient because only 1 feature
    assert np.isclose(lin_reg.slope, sk_lin_reg.coef_[0])
    assert np.isclose(lin_reg.intercept, sk_lin_reg.intercept_)

    # TODO: should probably add more test cases...

# task: write test_mysimplelinearregressor_predict()
def test_mysimplelinearregressor_predict():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    lin_reg = MySimpleLinearRegressor()
    lin_reg.fit(X_train, y_train)
    X_test = [[150], [175]]
    y_pred = lin_reg.predict(X_test)
    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    sk_y_pred = sk_lin_reg.predict(X_test)
    assert np.allclose(y_pred, sk_y_pred)

    # TODO: should probably add more test cases...

# test driven development (TDD)
# write the unit tests before the units themselves
# often makes writing the units go smoother because you
# have a deep understanding of what makes them "correct"