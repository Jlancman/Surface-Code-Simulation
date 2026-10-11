"""
analysis.py -- threshold extraction and sub-threshold scaling fits.

THRESHOLD (use TOTAL failure probability over d rounds -- dividing by the
round count d would bias the crossing upward): below p_th, increasing the code distance d suppresses logical errors;
above it, larger codes are worse.  So curves p_L(p) for different d cross at
p_th.  We locate the crossing of every neighbouring pair (d, d+2) by linear
interpolation of  log p_L(d+2) - log p_L(d)  between the two grid points where
it changes sign, then report the mean (and spread) over pairs.

SUB-THRESHOLD SCALING (Fowler et al. 2012; Google Quantum AI, Nature 614, 676 (2023)):
        p_L(d) ~ A * (p / p_th)^((d+1)/2)  =>  Lambda = p_L(d)/p_L(d+2) = (p_th/p)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def crossing(ps: np.ndarray, small: np.ndarray, large: np.ndarray):
    """p where large-d curve crosses small-d curve (None if no crossing)."""
    ok = (small > 0) & (large > 0)
    ps, small, large = ps[ok], small[ok], large[ok]
    diff = np.log(large) - np.log(small)       # <0 below threshold, >0 above
    for i in range(len(ps) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            f = -diff[i] / (diff[i + 1] - diff[i])
            # interpolate in log p (curves are ~power laws)
            return float(np.exp(np.log(ps[i]) + f * (np.log(ps[i + 1]) - np.log(ps[i]))))
    return None


def estimate_threshold(ps: List[float], curves: Dict[int, List[float]]) -> Tuple[float, float, Dict]:
    ds = sorted(curves)
    ps = np.asarray(ps)
    xs = {}
    for a, b in zip(ds[:-1], ds[1:]):
        c = crossing(ps, np.asarray(curves[a]), np.asarray(curves[b]))
        if c is not None:
            xs[(a, b)] = c
    if not xs:
        return float("nan"), float("nan"), xs
    v = np.array(list(xs.values()))
    return float(v.mean()), float(v.std()), xs


def fit_subthreshold(ps, pL, d, p_th, p_max_fraction=0.7):
    """Fit log pL = log A + k * log(p/p_th) using points with p < frac*p_th.
    Theory predicts k = (d+1)/2.  Returns (A, k)."""
    ps, pL = np.asarray(ps), np.asarray(pL)
    m = (ps < p_max_fraction * p_th) & (pL > 0)
    if m.sum() < 2:
        return float("nan"), float("nan")
    k, logA = np.polyfit(np.log(ps[m] / p_th), np.log(pL[m]), 1)
    return float(np.exp(logA)), float(k)
