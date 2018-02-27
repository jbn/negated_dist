import unittest
import numpy as np
import warnings
import contextlib

from scipy.stats import norm, binom, rv_histogram
from negated_dist import (_bind_to_attr,
                          _negated_apply,
                          _negated_return,
                          _negated_interval,
                          _negated_stats,
                          NegatedDist)


CONTINUOUS_DIST = norm()
DISCRETE_DIST = binom(5, 0.5)
HISTOGRAM_DIST = rv_histogram(np.histogram(CONTINUOUS_DIST.rvs(1000)))

DIST_PAIRS = [(d, NegatedDist(d))
              for d in [CONTINUOUS_DIST, DISCRETE_DIST, HISTOGRAM_DIST]]


@contextlib.contextmanager
def swollow_integration_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestNegatedDistHelpers(unittest.TestCase):

    def test_bind_to_attr(self):
        f = _bind_to_attr('mean', lambda g: g)(DISCRETE_DIST)
        self.assertEqual(f(), DISCRETE_DIST.mean())

    def test_negated_apply(self):
        negated = _negated_apply(DISCRETE_DIST.pmf)
        self.assertEqual(negated(-1), DISCRETE_DIST.pmf(1))

    def test_negated_return(self):
        negated = _negated_return(DISCRETE_DIST.mean)
        self.assertEqual(-negated(), DISCRETE_DIST.mean())

    def test_negated_interval(self):
        negated = _negated_interval(DISCRETE_DIST)
        self.assertEqual(DISCRETE_DIST.interval(1.0), (-1.0, 5.0))
        self.assertEqual(negated(1.0), (-5.0, 1.0))

    def test_negated_stats(self):
        negated = _negated_stats(DISCRETE_DIST)
        self.assertEqual(DISCRETE_DIST.stats(), (2.5, 1.25))
        self.assertEqual(negated(), (-2.5, 1.25))


class TestNegatedDist(unittest.TestCase):

    def test_args(self):
        for X, Y in DIST_PAIRS:
            # rv_histogram has no args.
            # I think this may be dangerous though.
            # This is the params of the underlying distribution, not the
            # negated one!
            if not hasattr(X, 'args'):
                continue
            self.assertEqual(X.args, Y.args)

    def test_central_tendency(self):
        for X, Y in DIST_PAIRS:
            self.assertEqual(X.mean(), -Y.mean())
            self.assertEqual(X.median(), -Y.median())
            with swollow_integration_warning():
                self.assertEqual(X.expect(), -Y.expect())

    def test_dispersion(self):
        for X, Y in DIST_PAIRS:
            self.assertEqual(X.std(), Y.std())
            self.assertEqual(X.entropy(), Y.entropy())
            self.assertEqual(X.var(), Y.var())

    def test_bounds(self):
        for X, Y in DIST_PAIRS:
            a, b = X.interval(0.9)
            c, d = Y.interval(0.9)
            self.assertEqual(a, -d)
            self.assertEqual(b, -c)
            self.assertEqual(X.a, -Y.b)
            self.assertEqual(X.b, -Y.a)

    def test_rvs(self):
        for X, Y in DIST_PAIRS:
            # WARNING: .rvs for rv_histogram has different interface!
            delta = abs(X.mean() + Y.rvs(size=1000).mean())
            self.assertLess(delta, 0.075, X)

    def test_stats(self):
        for X, Y in DIST_PAIRS:
            a, b = X.stats()
            c, d = Y.stats()
            self.assertEqual(a, -c)
            self.assertEqual(b, d)

    def test_pdf(self):
        for X, Y in DIST_PAIRS:
            try:
                self.assertEqual(X.pdf(X.mean()), Y.pdf(Y.mean()))
                self.assertEqual(X.logpdf(X.mean()), Y.logpdf(Y.mean()))
            except AttributeError:
                self.assertEqual(X.pmf(X.mean()), Y.pmf(Y.mean()))
                self.assertEqual(X.logpmf(X.mean()), Y.logpmf(Y.mean()))

    def test_cdf_and_sf(self):
        for X, Y in DIST_PAIRS:
            z = X.rvs()
            self.assertAlmostEqual(X.cdf(z), 1 - Y.cdf(-z))
            self.assertAlmostEqual(X.sf(z), 1 - Y.sf(-z))
            self.assertAlmostEqual(np.log(1 - X.cdf(z)), Y.logcdf(-z))
            self.assertAlmostEqual(np.log(1 - X.sf(z)), Y.logsf(-z))

    def test_cdf(self):
        for X, Y in DIST_PAIRS:
            z = X.rvs()
            self.assertAlmostEqual(X.cdf(z), 1 - Y.cdf(-z))

    def test_ppf_and_isf(self):
        for X, Y in DIST_PAIRS:
            z = np.random.rand()
            self.assertAlmostEqual(X.ppf(1 - z), -Y.ppf(z))
            self.assertAlmostEqual(X.isf(1 - z), -Y.isf(z))

    def test_negated_moment(self):
        X = norm(-1)
        Y = NegatedDist(norm(1))
        for n in range(6):
            self.assertAlmostEqual(X.moment(n), Y.moment(n), n)


if __name__ == '__main__':
    unittest.main()
