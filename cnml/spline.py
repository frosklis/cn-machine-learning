# Splines
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

    This is a monovariate spline regressor. It fits a curve using chained
    polynomials that are visually "pleasant". This is because the continuity
    of the second derivative is mantained.

    Because this is for fitting, not for interpolating there is no point in
    talking about natural splines or constrained splines.

    A nice feature as well is that the degree of the first and last
    polynomials can be tweaked so that it is, for example, a horizontal line
    (degree=0).

    .. math::
        ax^3 + bx^2 + cx + d = 0

        3ax^2 + 2bx + c = 0
        
        6ax + 2b = 0

        b = -3ax
        
        c = +3ax^2
        
        d = -ax^3

    Parameters
    ----------
    num_knots : int, default=3
        number of knots that will be used
    degrees : tuple(int, int), default=(3, 3)
        degree of the first and last polynomials
    degree : int, default=3
        degree of the spline, currently this parameter is ignored, it always
        uses the default value of 3
    """

    def __init__(self, num_knots: int = 3, degrees=(3, 3), degree=3):
        self.num_knots = num_knots
        self.degrees = degrees
        self.degree = 3

    def fit(self, X, y):
        """Fits the spline with the given parameters.

        Internally, in order to avoid numerical stability, the input values
        are scaled to fit in the [-1, -1] range.

        Parameters
            X ({array-like, sparse matrix} of shape (n_samples,
            n_features)): The training input samples. Internally, it will be
            converted to
                ``dtype=np.float32`` and if a sparse matrix is provided to a
                sparse ``csc_matrix``.
            y (array-like of shape (n_samples,)): The target values

        Returns:
            WoETransformer: **self** -- Fitted predictor.


        """
        degrees = self.degrees
        num_knots = self.num_knots
        degree = self.degree

        x_max, x_min = np.max(X), np.min(X)
        y_max, y_min = np.max(y), np.min(y)
        tX = (X - x_min) / (x_max - x_min)
        ty = (y - y_min) / (y_max - y_min)
        self._maxmin = x_max, x_min, y_max, y_min

        objective = lambda x: residuals(x, x=tX, y=ty, num_knots=num_knots,
                                        degrees=degrees)
        knots_0 = np.percentile(tX, np.linspace(0, 100, num_knots + 2))[1:-1]
        fit = np.polyfit(tX, ty, degree)
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
        self._knots = res.x[-num_knots:] * (x_max - x_min) + x_min

        return self

    def predict(self, X):
        x_max, x_min, y_max, y_min = self._maxmin
        tX = (X - x_min) / (x_max - x_min)
        ty = eval_spline(tX, self._coefs, num_knots=self.num_knots,
                         degrees=self.degrees)
        return ty * (y_max - y_min) + y_min

    def _debug(self):
        return _helper(self._coefs, num_knots=self.num_knots,
                       degrees=self.degrees)
