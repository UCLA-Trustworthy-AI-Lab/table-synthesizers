import numpy as np
import pandas as pd

try:
    from mbi import FactoredInference, Dataset, Domain, GraphicalModel
except ImportError:
    print("Please install mbi with:\n   pip install git+https://github.com/ryan112358/private-pgm.git")

import itertools
from mbi import Dataset, FactoredInference, Domain
from scipy import sparse


import itertools

import numpy as np
from scipy.special import softmax
from opendp.measurements import make_base_laplace, make_base_gaussian
from opendp.mod import enable_features, binary_search_param
from opendp.combinators import make_zCDP_to_approxDP, make_fix_delta

prng = np.random


def exponential_mechanism(qualities, epsilon, sensitivity=1.0, base_measure=None):
    if isinstance(qualities, dict):
        keys = list(qualities.keys())
        qualities = np.array([qualities[key] for key in keys])
        if base_measure is not None:
            base_measure = np.log([base_measure[key] for key in keys])
    else:
        qualities = np.array(qualities)
        keys = np.arange(qualities.size)

    """ Sample a candidate from the permute-and-flip mechanism """
    q = qualities - qualities.max()
    if base_measure is None:
        p = softmax(0.5 * epsilon / sensitivity * q)
    else:
        p = softmax(0.5 * epsilon / sensitivity * q + base_measure)

    return keys[prng.choice(p.size, p=p)]

def gaussian_noise(sigma, size=None):
    enable_features('floating-point', 'contrib')
    meas = make_base_gaussian(sigma)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def laplace_noise(scale, size=None):
    enable_features('floating-point', 'contrib')
    meas = make_base_laplace(scale)
    if size is None:
        return meas(0.0)
    else:
        return [meas(0.0) for _ in range(size)]

def cdp_rho(epsilon, delta):
    budget = (epsilon, delta)
    enable_features('floating-point', 'contrib')
    def make_fixed_approxDP_gaussian(scale):
        adp = make_zCDP_to_approxDP(make_base_gaussian(scale))
        return make_fix_delta(adp, delta=budget[1])
    scale = binary_search_param(
        make_fixed_approxDP_gaussian,
        d_in=1.0, d_out=budget, T=float)
    return make_base_gaussian(scale).map(1.)

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


class Identity(sparse.linalg.LinearOperator):
    def __init__(self, n):
        self.shape = (n,n)
        self.dtype = np.float64
    def _matmat(self, X):
        return X
    def __matmul__(self, X):
        return X
    def _transpose(self):
        return self
    def _adjoint(self):
        return self

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2 ** 20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)

    return {cl: score(cl) for cl in downward_closure(workload)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans