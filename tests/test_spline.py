import numpy as np
import pytest

from cnml.spline import Spline


@pytest.mark.parametrize(
    "offset", [0, -1000, -500, -100, -50, -5,
               1000, 500, 100, 50, 5, ]
)
@pytest.mark.parametrize('allowed_error', [0.1])
def test_numerical_stability(offset, allowed_error):
    X = np.linspace(0, 6.28, 1000) + offset
    y = np.sin(X)
    model = Spline().fit(X, y)
    y_hat = model.predict(X)

    error = np.abs(y - y_hat)
    assert error.max() < allowed_error


@pytest.mark.parametrize('degree_first', [0, 1, 2, 3])
@pytest.mark.parametrize('degree_last', [0, 1, 2, 3])
@pytest.mark.parametrize('num_knots', [0, 1, 2, 3, 4])
def test_parameters(degree_first, degree_last, num_knots):
    X = np.linspace(0, 6.28, 100)
    y = np.sin(X)
    model = Spline(degrees=(degree_first, degree_last),
                   num_knots=num_knots).fit(
        X, y)
    y_hat = model.predict(X)
    assert np.isnan(y_hat).sum() == 0, "there are null values in the output"
    assert len(y_hat) == len(y), "correct number of data points"
