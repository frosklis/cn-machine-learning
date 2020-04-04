import numpy as np

from cnml.spline import Spline


def test_sin():
    X = np.linspace(0, 6.28, 1000)
    y = np.sin(X)
    model = Spline().fit(X, y)
    y_hat = model.predict(X)

    error = np.abs(y - y_hat)
    assert error.max() < .1


def test_sin_stable():
    X = np.linspace(0+10000, 6.28+10000, 1000)
    y = np.sin(X)
    model = Spline().fit(X, y)
    y_hat = model.predict(X)

    error = np.abs(y - y_hat)
    assert error.max() < .1



def test_sin_0():
    X = np.linspace(0, 6.28, 1000)
    y = np.sin(X)
    model = Spline(degrees=(3, 1), num_knots=4).fit(X, y)
    y_hat = model.predict(X)

    error = np.abs(y - y_hat)
    assert error.max() < .1

def test_grade_2():
    X = np.linspace(0, 6.28, 1000)
    y = np.sin(X)
    model = Spline(degrees=(2, 2), num_knots=5).fit(X, y)
    y_hat = model.predict(X)

    error = np.abs(y - y_hat)
    assert error.max() < .1

def test_no_knots():
    X = np.linspace(0, 6.28, 1000)
    y = np.sin(X)
    model = Spline(num_knots=0).fit(X, y)
    y_hat = model.predict(X)
    assert np.isnan(y_hat).sum() == 0, "there are null values in the output"
    assert len(y_hat) == len(y), "correct number of data points"
