"""
experiments.py -- the three experiments requested.

 1. threshold_sweep       : p_L vs p for several distances (uniform noise)
 2. heterogeneity_sweep   : informed vs mismatched decoder on non-uniform noise
 3. detection_map         : detector firing probabilities on a hot-spot device
Each returns plain Python/NumPy data so plotting is decoupled.
"""
from __future__ import annotations

from multiprocessing import Pool
from typing import Callable, Dict, List

import numpy as np

import noise as nz
from circuits import build_circuit
from decoding import run_memory, per_round, detection_rates


# ---------------------------------------------------------------- 1. threshold
def _thr_task(args):
    d, p, level, shots, fails, seed = args
    c = build_circuit(d, p, level)
    r = run_memory(c, max_shots=shots, max_failures=fails, seed=seed)
    rounds = 1 if level == "code_capacity" else d
    return d, p, r.p_L, r.stderr, per_round(r.p_L, rounds), r.shots, r.failures


def threshold_sweep(distances: List[int], ps: List[float], level: str = "circuit",
                    max_shots: int = 100_000, max_failures: int = 300,
                    workers: int = 2) -> Dict:
    tasks = [(d, p, level, max_shots, max_failures, 1000 * d + i)
             for d in distances for i, p in enumerate(ps)]
    with Pool(workers) as pool:
        rows = pool.map(_thr_task, tasks, chunksize=1)
    out = {"level": level, "ps": list(ps), "distances": list(distances),
           "p_L": {d: [] for d in distances}, "err": {d: [] for d in distances},
           "p_round": {d: [] for d in distances}}
    for d, p, pl, se, pr, sh, fl in rows:
        out["p_L"][d].append(pl); out["err"][d].append(se); out["p_round"][d].append(pr)
    return out


# ------------------------------------------------------- 2. non-uniform noise
def _het_task(args):
    d, p, kind, param, seed, shots, fails = args
    base = build_circuit(d, p, "circuit")
    if kind == "lognormal":
        prof = nz.lognormal_profile(base, param, seed=seed)
    elif kind == "hotspot":
        prof = nz.hotspot_profile(base, n_bad=int(param[0]), factor=param[1], seed=seed)
    elif kind == "gradient":
        prof = nz.gradient_profile(base, param)
    else:
        prof = nz.uniform_profile(base)
    true_c = nz.apply_profile(base, prof)       # what the hardware really does
    # decoder A: knows every qubit's rate (DEM from the true circuit)
    informed = run_memory(true_c, true_c, shots, fails, seed=seed)
    # decoder B: assumes uniform noise at the same *average* rate
    uni_c = build_circuit(d, p, "circuit")
    mismatched = run_memory(true_c, uni_c, shots, fails, seed=seed)
    return (informed.p_L, informed.stderr, mismatched.p_L, mismatched.stderr)


def heterogeneity_sweep(d: int, p: float, kind: str, params: list,
                        n_disorder: int = 4, max_shots: int = 60_000,
                        max_failures: int = 300, workers: int = 2) -> Dict:
    """For each disorder strength, average over `n_disorder` random device
    realisations (different seeds -> different 'bad qubit' placements)."""
    tasks = [(d, p, kind, prm, s, max_shots, max_failures)
             for prm in params for s in range(n_disorder)]
    with Pool(workers) as pool:
        rows = pool.map(_het_task, tasks, chunksize=1)
    rows = np.array(rows).reshape(len(params), n_disorder, 4)
    return {"kind": kind, "d": d, "p": p, "params": params,
            "informed": rows[:, :, 0].mean(1), "informed_sd": rows[:, :, 0].std(1),
            "mismatched": rows[:, :, 2].mean(1), "mismatched_sd": rows[:, :, 2].std(1),
            "raw": rows}


# ---------------------------------------------------------- 3. detection map
def detection_map(d: int, p: float, n_bad: int, factor: float, seed: int = 3,
                  shots: int = 30_000) -> Dict:
    base = build_circuit(d, p, "circuit")
    prof = nz.hotspot_profile(base, n_bad, factor, seed=seed, data_only=True)
    true_c = nz.apply_profile(base, prof)
    return {"profile": prof,
            "qubit_rates_true": nz.qubit_error_rates(true_c),
            "det_true": detection_rates(true_c, shots),
            "det_uniform": detection_rates(base, shots),
            "d": d, "p": p}
