import numpy as np
from typing import Dict
from multiprocessing import Pool
from circuits import build_circuit, rounds_for
from decoding_and_sampling import run_memory, detection_rates, per_round, SHOTS, MAX_FAILURES
from noise import make_profile, apply_profile

WORKERS = 2                    # parallel processes for sweeps
DEVICE_SAMPLES = 4            # random chips averaged in the decoder comparison

#Experiments:
def thr_task(args):
    d, p, level, profile, shots, fails, seed = args
    c = build_circuit(d, p, level, profile)
    r = run_memory(c, max_shots=shots, max_failures=fails, seed=seed)
    return d, r.p_L, r.stderr, per_round(r.p_L, rounds_for(level, d))

def threshold_sweep(distances, ps, level="circuit", profile=None,
                    shots=SHOTS, fails=MAX_FAILURES, workers=WORKERS) -> Dict:
    """Logical failure vs physical error rate for several code sizes."""
    tasks = [(d, p, level, profile, shots, fails, 1000 * d + i)
             for d in distances for i, p in enumerate(ps)]
    with Pool(workers) as pool:
        rows = pool.map(thr_task, tasks, chunksize=1)
    out = {"level": level, "ps": list(ps), "distances": list(distances),
           "p_L": {d: [] for d in distances}, "err": {d: [] for d in distances},
           "p_round": {d: [] for d in distances}}
    for d, pl, se, pr in rows:
        out["p_L"][d].append(pl); out["err"][d].append(se); out["p_round"][d].append(pr)
    return out


def het_task(args):
    d, p, kind, param, seed, shots, fails = args
    true_c = build_circuit(d, p, "circuit", (kind, param), seed=seed)  # what the chip does
    uni_c = build_circuit(d, p, "circuit")                              # what a naive decoder assumes
    informed = run_memory(true_c, true_c, shots, fails, seed=seed)
    mismatched = run_memory(true_c, uni_c, shots, fails, seed=seed)
    return informed.p_L, informed.stderr, mismatched.p_L, mismatched.stderr


def decoder_comparison(d, p, kind, params, n_devices=DEVICE_SAMPLES,
                       shots=SHOTS, fails=MAX_FAILURES, workers=WORKERS) -> Dict:
    """Informed vs mismatched decoder for increasing unevenness, averaged over
    `n_devices` random chips (different bad-qubit placements)."""
    tasks = [(d, p, kind, prm, s, shots, fails) for prm in params for s in range(n_devices)]
    with Pool(workers) as pool:
        rows = pool.map(het_task, tasks, chunksize=1)
    rows = np.array(rows).reshape(len(params), n_devices, 4)
    return {"kind": kind, "d": d, "p": p, "params": params,
            "informed": rows[:, :, 0].mean(1), "informed_sd": rows[:, :, 0].std(1),
            "mismatched": rows[:, :, 2].mean(1), "mismatched_sd": rows[:, :, 2].std(1)}


def detection_map(d, p, kind, param, seed=3, shots=SHOTS) -> Dict:
    """Detector firing probabilities on a non-uniform chip vs a uniform one."""
    base = build_circuit(d, p, "circuit")
    prof = make_profile(base, kind, param, seed=seed, data_only=(kind == "hotspot"))
    true_c = apply_profile(base, prof)
    return {"profile": prof, "det_true": detection_rates(true_c, shots),
            "det_uniform": detection_rates(base, shots), "d": d, "p": p}
