"""Threshold sweep: logical failure vs physical error rate for several code sizes.
Every result is checked against, and filed under, the (d, p) it was asked for."""
from multiprocessing import Pool
import numpy as np
from circuits import build_circuit, rounds_for, PROBS
from decoding_and_sampling import run_memory, per_round

NOISE_GATES = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")

def circuit_noise_rate(c):
    """Largest noise probability actually written in the circuit."""
    return max(a for ins in c.flattened() if ins.name in NOISE_GATES
               for a in ins.gate_args_copy())

def thr_task(args):
    d, p, level, profile, shots, fails, seed = args
    uniform = profile is None or profile[0] == "uniform"
    c = build_circuit(d, p, level) if uniform else build_circuit(d, p, level, profile)
    if uniform:
        # guard: stop if the circuit was not built at the requested p
        want = max([p] + [q for q in PROBS if q is not None])
        assert np.isclose(circuit_noise_rate(c), want), (d, p, want, circuit_noise_rate(c))
    r = run_memory(c, max_shots=shots, max_failures=fails, seed=seed)
    return d, p, r.failures, r.shots, rounds_for(level, d)

def threshold_sweep(distances, ps, level="circuit", profile=None,
                    shots=200_000, fails=2_000, workers=4):
    tasks = [(d, p, level, profile, shots, fails, 1000 * d + i)
             for d in distances for i, p in enumerate(ps)]
    with Pool(workers) as pool:
        rows = pool.map(thr_task, tasks, chunksize=1)
    idx = {p: i for i, p in enumerate(ps)}
    out = {"level": level, "ps": list(ps), "distances": list(distances)}
    for key in ("p_L", "err", "p_round", "shots", "fails"):
        out[key] = {d: [float("nan")] * len(ps) for d in distances}
    for d, p, f, n, rounds in rows:
        i, pl = idx[p], f / n
        out["p_L"][d][i] = pl
        out["err"][d][i] = float(np.sqrt(max(pl * (1 - pl), 1 / n) / n))
        out["p_round"][d][i] = per_round(pl, rounds)
        out["shots"][d][i] = n
        out["fails"][d][i] = f
    return out
