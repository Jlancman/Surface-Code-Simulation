"""
run_all.py -- run every experiment, save figures + JSON to ./results

    python run_all.py --quick      # ~ few minutes, noisier curves
    python run_all.py              # default, tighter statistics
"""
import argparse, json, os, time
import numpy as np

import experiments as ex
import plotting as pl
from analysis import estimate_threshold, fit_subthreshold

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
summary = {}

def log(msg): print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

#Thresholds for the three noise levels:
configs = {
    "circuit":          dict(d=[3, 5, 7, 9],
                             ps=list(np.linspace(0.006, 0.016, 9))),
    "phenomenological": dict(d=[5, 7, 9],
                             ps=list(np.linspace(0.02, 0.05, 9))),
    "code_capacity":    dict(d=[3, 5, 7, 9],
                             ps=list(np.linspace(0.08, 0.20, 9))),
}
shots = 400_000
fails = 1000
for level, cfg in configs.items():
    log(f"threshold sweep: {level}")
    res = ex.threshold_sweep(cfg["d"], cfg["ps"], level, shots, fails, a.workers)
    th, sd, pairs = estimate_threshold(res["ps"], res["p_L"])
    log(f"   p_th = {th:.4f} +/- {sd:.4f}   pairs={ {k: round(v,4) for k,v in pairs.items()} }")
    pl.plot_threshold(res, th, sd, f"{a.out}/threshold_{level}.png", key="p_L")
    summary[level] = {"p_th": th, "p_th_largest_pair": pairs[max(pairs)] if pairs else None, "p_th_sd": sd, "pairs": {str(k): v for k, v in pairs.items()},
                      "data": {"ps": res["ps"], "p_round": {str(d): v for d, v in res["p_round"].items()},
                               "p_L": {str(d): v for d, v in res["p_L"].items()}}}
    if level == "circuit":
        # logical-vs-physical figure uses a wider low-p range to show suppression
        lo_ps = list(np.geomspace(0.001, 0.012, 11))
        res2 = ex.threshold_sweep(cfg["d"], lo_ps, "circuit", shots, fails, a.workers)
        th2, sd2, _ = estimate_threshold(res2["ps"], res2["p_L"])
        th_use = th if np.isfinite(th) else th2
        pl.plot_logical_vs_physical(res2, th_use, f"{a.out}/logical_vs_physical.png")
        pl.plot_threshold(res2, th2, sd2, f"{a.out}/threshold_circuit_wide.png", key="p_L")
        summary["circuit_wide"] = {"p_th": th2, "ps": res2["ps"],
                                   "p_round": {str(d): v for d, v in res2["p_round"].items()},
                                   "fits": {str(d): fit_subthreshold(res2["ps"], res2["p_round"][d], d, th_use)
                                            for d in res2["distances"]}}

# ---- 2. non-uniform noise, informed vs mismatched decoder -------------------
dH, pH = 5, (0.004)
nd = 12
hs = 200_000
log("heterogeneity: log-normal spread")
r1 = ex.heterogeneity_sweep(dH, pH, "lognormal", [0.0, 0.5, 1.0, 1.5], nd, hs, 300, a.workers)
pl.plot_heterogeneity(r1, f"{a.out}/nonuniform_lognormal.png", "log-normal spread $\\sigma$ of qubit error rates")
log("heterogeneity: hot qubits")
r2 = ex.heterogeneity_sweep(dH, pH, "hotspot", [(0, 1), (3, 10), (3, 20), (3, 40)][:4], nd, hs, 300, a.workers)
pl.plot_heterogeneity(r2, f"{a.out}/nonuniform_hotspot.png", "error-rate factor of 3 hot qubits (x nominal)")
for r, nm in ((r1, "lognormal"), (r2, "hotspot")):
    summary["het_" + nm] = {"params": [list(p) if isinstance(p, tuple) else p for p in r["params"]],
                            "informed": r["informed"].tolist(), "mismatched": r["mismatched"].tolist(),
                            "informed_sd": r["informed_sd"].tolist(), "mismatched_sd": r["mismatched_sd"].tolist()}
    log(f"   {nm}: informed={np.round(r['informed'],4)} mismatched={np.round(r['mismatched'],4)}")

# ---- 3. syndrome detection map ---------------------------------------------
log("detection map")
dm = ex.detection_map(7, 0.003, n_bad=3, factor=15.0)
pl.plot_detection_map(dm, f"{a.out}/detection_map.png")

json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
log("done")
