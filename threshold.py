import numpy as np
from typing import Dict, List

"""
Definitions (for readability and reference):
- p_th: threshold physical error rate. Below it, a bigger code (larger d) gives a
lower logical error rate; above it, a bigger code is worse (more places for errors).
- P_L: probability of logical error. Below threshold P_L ~ A (p/p_th)^((d+1)/2):
an uncorrectable error needs a chain of about (d+1)/2 faults across the code.
- lambda: suppression factor P_L(d)/P_L(d+2), roughly p_th/p.
"""
#Threshold analysis
def _fit_curve(logp, logpl, w):
    """Weighted quadratic fit of log P_L vs log p (smooth through the noise)."""
    return np.polyfit(logp, logpl, 2, w=w)

def crossing(ps, small, large, n_small=None, n_large=None):
    """p where the larger-d curve crosses the smaller-d one (None if no crossing).
    Each curve is fitted with a smooth quadratic in log-log space (weighted by
    1/stderr when shot counts are given), and the root of the difference is
    taken. Using fits instead of the first sign change of raw points stops
    one noisy point from moving the answer."""
    ps, small, large = map(np.asarray, (ps, small, large))
    ok = (small > 0) & (large > 0)
    if ok.sum() < 3:
        return None
    lp = np.log(ps[ok])
    def weights(pl, n):
        if n is None:
            return None
        n = np.asarray(n)[ok]
        return np.sqrt(n * pl / (1 - pl))      # 1 / stderr of log p_L
    a = _fit_curve(lp, np.log(small[ok]), weights(small[ok], n_small))
    b = _fit_curve(lp, np.log(large[ok]), weights(large[ok], n_large))
    roots = np.roots(b - a)
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= lp.min()) & (roots <= lp.max())]
    # keep the root where the larger code goes from better to worse
    roots = [r for r in roots if np.polyval(np.polyder(b - a), r) > 0]
    return float(np.exp(roots[0])) if roots else None

def estimate_threshold(ps, curves: Dict[int, List[float]], shots=None,
                       n_boot=300, seed=0):
    """Crossing of each neighbouring distance pair, with a bootstrap error bar.
    The error bar comes from resampling every point's failure count from
    Binomial(shots, p_L), i.e. it is the statistical (shot-noise) uncertainty.
    Crossings drift with d (finite-size effects), so the largest pair is the
    best estimate of the true threshold; that is what is returned as p_th."""
    ds = sorted(curves)
    rng = np.random.default_rng(seed)
    xs, errs = {}, {}
    for a, b in zip(ds[:-1], ds[1:]):
        na = None if shots is None else shots[a]
        nb = None if shots is None else shots[b]
        c = crossing(ps, curves[a], curves[b], na, nb)
        if c is None:
            continue
        xs[(a, b)] = c
        if shots is not None:
            boot = []
            for _ in range(n_boot):
                ra = rng.binomial(na, curves[a]) / np.asarray(na)
                rb = rng.binomial(nb, curves[b]) / np.asarray(nb)
                cb = crossing(ps, ra, rb, na, nb)
                if cb is not None:
                    boot.append(cb)
            errs[(a, b)] = float(np.std(boot)) if boot else float("nan")
    if not xs:
        return float("nan"), float("nan"), xs
    last = max(xs)
    return xs[last], errs.get(last, float("nan")), xs

def fit_subthreshold(ps, pL, p_th, fails=None, p_max_fraction=0.5, min_failures=20):
    """Fit p_L ~ A (p/p_th)^k on points well below threshold that have enough
    failures to be trusted. Theory (asymptotically, p -> 0): k = (d+1)/2."""
    ps, pL = np.asarray(ps), np.asarray(pL)
    m = (ps < p_max_fraction * p_th) & (pL > 0)
    if fails is not None:
        m &= np.asarray(fails) >= min_failures
    if m.sum() < 3:
        return float("nan"), float("nan")
    k, logA = np.polyfit(np.log(ps[m] / p_th), np.log(pL[m]), 1)
    return float(np.exp(logA)), float(k)
