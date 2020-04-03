# Splines

"""
ax^3 + bx^2 + cx + d = 0
3ax^2 + 2bx + c = 0
6ax + 2b = 0

b = -3ax
c = +3ax^2
d = -ax^3
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


def _helper(coefs: np.array, num_knots: int = 3, degrees=(3, 3),
            clip=(-np.inf, np.inf)):
    degree = 3
    first, last = degrees
    assert len(
        coefs) == degree + 1 + num_knots * 2, "wrong number of coefficients"
    first_poly = coefs[:degree + 1]

    # No knots, trick into defining on only knot at +infinity
    if num_knots == 0:
        if first < degree:
            first_poly[0:3 - first] = 0
        knots = np.array([np.inf])
        polynomials = np.array([first_poly])
        return knots, polynomials

    rest = coefs[degree + 1:degree + 1 + num_knots]
    knots = np.clip(np.sort(coefs[degree + 1 + num_knots:]), *clip)
    coefs[degree + 1 + num_knots:] = knots

    def calculate_polynomials():
        shifted = np.roll(rest, 1)
        shifted[0] = coefs[0]
        delta_a = rest - shifted
        deltas = np.array([
            delta_a,
            -3 * delta_a * knots,
            +3 * delta_a * knots ** 2,
            -1 * delta_a * knots ** 3,
        ]).T
        polynomials = np.cumsum(
            np.concatenate([[first_poly], deltas], axis=0),
            axis=0)

        return polynomials

    polynomials = calculate_polynomials()

    # Now tune the polynomials
    for idx, idx2, new_degree in zip([0, -1], [1, -2], degrees):
        if new_degree == degree:
            continue
        poly = polynomials[idx]  # copied by reference on purpose
        poly2 = polynomials[idx2]  # copied by reference on purpose
        knot = knots[idx]
        knot2 = knots[idx2]
        f, df, df2 = [np.polyval(np.polyder(poly, i), knot)
                      for i in range(degree)]
        g, dg, dg2 = [np.polyval(np.polyder(poly2, i), knot2)
                      for i in range(degree)]
        if new_degree == 2 and poly[0] != 0:
            poly[0] = 0
            poly[1] = df2 / 2
            poly[2] = df - df2 * knot
            poly[3] = f + df2 * knot ** 2 / 2 - df * knot
        else:
            poly[0] = 0
            poly[1] = 0
            poly[2] = df if new_degree == 1 else 0
            # solve a linear system K * p = b
            k = np.array([
                # [knot ** 3, knot ** 2, knot, 1],
                [3 * knot ** 2, 2 * knot, 1, 0],
                # [6 * knot, 2, 0, 0],
                [knot2 ** 3, knot2 ** 2, knot2, 1],
                [3 * knot2 ** 2, 2 * knot2, 1, 0],
                [6 * knot2, 2, 0, 0],
            ])
            b = np.array([
                # f,
                df if new_degree == 1 else 0,
                # 0,
                g,
                dg,
                dg2
            ])
            solution = np.linalg.lstsq(k, b, rcond=None)
            poly2[:] = solution[0]
            poly2[3] += g - np.polyval(poly2, knot2)
            poly[3] += np.polyval(poly2, knot) - np.polyval(poly, knot)

    return knots, polynomials


def eval_spline(x, coefs: np.array, num_knots: int = 3, degrees=(3, 3)):
    knots, polynomials = _helper(coefs, num_knots=num_knots, degrees=degrees)
    pred_y = x.copy()

    for i in range(num_knots + 1):
        if i == 0:
            indices = np.argwhere(x < knots[i])
        elif i == num_knots:
            indices = np.argwhere(x >= knots[i - 1])
        else:
            indices = np.argwhere(
                (x >= knots[i - 1]) &
                (x < knots[i]))
        # Common case
        pred_y[indices] = np.polyval(polynomials[i], x[indices])

    return pred_y


def residuals(coefs: np.array, num_knots: int = 3, x=None,
              y=None, degrees=(3, 3)):
    knots, polynomials = _helper(coefs, num_knots=num_knots,
                                 degrees=degrees, clip=(x.min(), x.max()))
    residual = 0
    for i in range(num_knots + 1):
        if i == 0:
            indices = np.argwhere(x < knots[i])
        elif i == num_knots:
            indices = np.argwhere(x >= knots[i - 1])
        else:
            indices = np.argwhere(
                (x >= knots[i - 1]) &
                (x < knots[i]))
        # Common case
        pred_y = np.polyval(polynomials[i], x[indices])
        real_y = y[indices]

        residual += np.sum((pred_y - real_y) ** 2)

    return residual


class Spline(BaseEstimator):
    """Cubic splines regressor


    """

    def fit(self, X, y, num_knots=3, degrees=(3, 3)):
        self._num_knots = num_knots
        self._degrees = degrees
        degree = 3
        objective = lambda x: residuals(x, x=X, y=y, num_knots=num_knots,
                                        degrees=degrees)
        knots_0 = np.percentile(X, np.linspace(0, 100, num_knots + 2))[1:-1]
        fit = np.polyfit(X, y, degree)
        zeros = np.zeros(num_knots) + fit[0]

        # fit = np.zeros(degree+1)
        # 'Powell'
        # 'Nelder-Mead'
        # 'BFGS'
        n = len(X)
        res = minimize(
            objective,
            np.concatenate([fit, zeros, knots_0]),
            method='Nelder-Mead',
            options={'maxfev': int(np.sqrt(n) * 200)}
        )
        res = minimize(
            objective,
            res.x,
            method='BFGS'
        )
        res = minimize(
            objective,
            res.x,
            method='Powell'
        )
        self._coefs = res.x
        self._optimization_result = res
        k = degree + 1 + num_knots * 2
        self._bic = n * np.log(res.fun / (n - 1)) + k * np.log(n)
        self._aic = 2 * k + n * np.log(res.fun / (n - 1))
        self._aic += 2 * k * (1 + k) / (n - k - 1)
        self._knots = res.x[-num_knots:]
        return self

    def predict(self, X):
        return eval_spline(X, self._coefs, num_knots=self._num_knots,
                           degrees=self._degrees)

    def _debug(self):
        return _helper(self._coefs, num_knots=self._num_knots,
                       degrees=self._degrees)
