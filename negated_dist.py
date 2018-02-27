import copy
import numpy as np


NEGATED_APPLY_METHS = {'pdf', 'pmf', 'logpmf', 'logpdf'}
NEGATED_RETURN_METHS = {'rvs', 'mean', 'median', 'expect'}
PASS_THROUGH_METHS = {'std', 'var', 'entropy', 'random_state', 'dist', 'args', 'kwds'}
SPECIAL_METHS = {'interval', 'a', 'b', 'stats', 'cdf', 'sf', 'logcdf', 'logsf'}


def _bind_to_attr(k, f):
    def _f(dist):
        return f(getattr(dist, k))
    return _f


def _negated_apply(f):
    def _f(x, *args, **kwargs):
        return f(-x, *args, **kwargs)
    return _f


def _negated_return(f):
    def _f(*args, **kwargs):
        return -f(*args, **kwargs)
    return _f


def _negated_interval(dist):
    def _f(*args, **kwargs):
        a, b = dist.interval(*args, **kwargs)
        return -b, -a
    return _f


def _negated_stats(dist):
    def _f(*args, **kwargs):
        mean, var = dist.stats(*args, **kwargs)
        return -mean, var
    return _f


def _negated_moment(dist):
    def _f(n, *args, **kwargs):
        return (-1)**(n) * dist.moment(n, *args, **kwargs)
    return _f


def _build_remapping():
    d = {}

    for k in NEGATED_APPLY_METHS:
        d[k] = _bind_to_attr(k, _negated_apply)

    for k in NEGATED_RETURN_METHS:
        d[k] = _bind_to_attr(k, _negated_return)

    for k in PASS_THROUGH_METHS:
        d[k] = _bind_to_attr(k, lambda f: f)

    d['interval'] = _negated_interval
    d['stats'] = _negated_stats
    d['cdf'] = _bind_to_attr('sf', _negated_apply)
    d['logcdf'] = _bind_to_attr('logsf', _negated_apply)
    d['sf'] = _bind_to_attr('cdf', _negated_apply)
    d['logsf'] = _bind_to_attr('logcdf', _negated_apply)
    d['ppf'] = _bind_to_attr('isf', _negated_return)
    d['isf'] = _bind_to_attr('ppf', _negated_return)
    d['moment'] = _negated_moment

    return d


REMAPPING = _build_remapping()


class NegatedDist:

    def __init__(self, dist):
        self.dist = dist

    @property
    def a(self):
        return -self.dist.b

    @property
    def b(self):
        return -self.dist.a

    def __getattr__(self, name):
        f = REMAPPING.get(name)

        if f is None:
            fmt = "{} not implemented in NegatedDist"
            raise NotImplementedError(fmt.format(name))
        else:
            return f(self.dist)

    def __deepcopy__(self, memo):
        return NegatedDist(copy.deepcopy(self.dist, memo))
