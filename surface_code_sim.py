"""
Surface code quantum memory simulator, using Stim and PyMatching.

Run it and pick a noise model from the menu:

    python surface_code_sim.py              # interactive menu
    python surface_code_sim.py --all        # run every experiment, save results/
    python surface_code_sim.py --selftest   # quick correctness checks

Stim generates a rotated surface-code memory circuit with one reference
error rate p, at one of three noise levels.
For non-uniform hardware, every noise instruction in the circuit is
rewritten so each qubit gets its own rate p * multiplier(qubit).
Detectors and observables are untouched.
Stim samples detection events; PyMatching decodes them using a matching
graph built from the circuit's detector error model. A shot fails when the
decoder's predicted logical flip differs from the true one.
Gate errors are specified in circuits.py, whereas running simulations here sweeps through error
probabilities for qubits in different forms. 
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import stim
from plotting_functions import (agg, plot_threshold, plot_logical_vs_physical, 
                                plot_decoder_comparison, plot_detection_map, plot_footprint)
from threshold import estimate_threshold
from noise import (NOISE_MODELS, PROFILE_PARAMS, make_profile, apply_profile, qubit_error_rates)
from circuits import build_circuit, rounds_for
from decoding_and_sampling import run_memory, per_round, SHOTS, MAX_FAILURES
from experiments import decoder_comparison, detection_map, het_task, DEVICE_SAMPLES
from sweep import threshold_sweep

#Settings
DISTANCE = 5                  # code size d (d x d data qubits); odd, >= 3
SWEEP_DISTANCES = [3, 5, 7]   # sizes compared in a threshold sweep
SWEEP_POINTS = 5             # error rates per threshold sweep (log-spaced)
SWEEP_SHOTS = 1000000        # max shots per sweep point
SWEEP_FAILURES = 5000        # stop a sweep point once this many failures are seen (~2% error)
# p ranges for the threshold sweeps, centred on each threshold; other models use their own "sweep"
THRESHOLD_RANGES = {"code_capacity": (0.004, 0.009), "phenomenological": (0.1, 0.5),
                    "circuit": (0.1, 0.5)}
EXACT_PS = {"repetition": [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5], 
            "circuit": [0.008, 0.009, 0.01, 0.011, 0.012]} # exact points for repetition code (Stim tutorial)
PER_ROUND = True #do you plot using logical error rate or error rate per round
OUTPUT_DIR = "C:\\Users\\jlanc\\oneDrive\\Desktop\\Qarakal Project\\surface_code_sim\\Results\\"      # where plots and summary.json are written
#[n, k, d] = d**2, 1, d

#User Interface:
def ask_choice(title, options):
    """options: list of (label, description or ''); returns 0-based index."""
    print(f"\n{title}")
    for i, (label, desc) in enumerate(options, 1):
        print(f"  {i}) {label}")
        if desc:
            print(f"       {desc}")
    while True:
        s = input(f"Choose 1-{len(options)}: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(options):
            return int(s) - 1
        print("Please type one of the numbers above.")

def ask_number(prompt, default, cast=float):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            return cast(s)
        except ValueError:
            print("  Please type a number, or press Enter for the default.")

def ask_profile_param(kind):
    info = PROFILE_PARAMS[kind]
    if kind == "hotspot":
        n, f = info["default"]
        return (n, ask_number(info["prompt"], f))
    if info["default"] is None:
        return None
    return ask_number(info["prompt"], info["default"])

def run_single(model, d, p, param):
    level, kind = model["level"], model["profile"]
    rounds = rounds_for(level, d)
    print(f"\nSimulating d={d}, p={p}, {rounds} round(s), up to {SHOTS:,} shots ...")
    t0 = time.time()
    if kind == "uniform":
        r = run_memory(build_circuit(d, p, level), seed=1)
        print(f"  logical failure rate : {r.p_L:.5f} +/- {r.stderr:.5f}  "
              f"({r.failures} failures in {r.shots:,} shots)")
        print(f"  per round            : {per_round(r.p_L, rounds):.5f}")
    else:
        inf_p, inf_e, mis_p, mis_e = het_task((d, p, kind, param, 0, SHOTS, MAX_FAILURES))
        print(f"  decoder knows each qubit's rate : {inf_p:.5f} +/- {inf_e:.5f}")
        print(f"  decoder assumes uniform noise   : {mis_p:.5f} +/- {mis_e:.5f}")
    print(f"  ({time.time() - t0:.1f}s)")

def sweep_ps(model):
    if model["key"] in EXACT_PS: 
        ps = EXACT_PS[model["key"]]
        return ps[0], ps[-1], ps
    lo, hi = THRESHOLD_RANGES.get(model["key"], model["sweep"])
    return lo, hi, list(np.geomspace(lo, hi, SWEEP_POINTS))

def run_threshold(model, param):
    lo, hi, ps = sweep_ps(model)
    profile = (model["profile"], param)
    print(f"\nThreshold sweep: d = {SWEEP_DISTANCES}, p from {lo} to {hi} ({SWEEP_POINTS} points), "
          f"up to {SWEEP_SHOTS:,} shots each ...")
    res = threshold_sweep(SWEEP_DISTANCES, ps, model["level"], profile,
                          shots=SWEEP_SHOTS, fails=SWEEP_FAILURES)
    if PER_ROUND: 
        res["p_L"] = res["p_round"]
        res["err"] = {d: [e / rounds_for(model["level"], d)
                           for e in res["err"][d]] for d in SWEEP_DISTANCES}
    th, sd, pairs = estimate_threshold(res["ps"], res["p_round"], res["shots"])
    print("\n  p        " + "".join(f"d={d:<9}" for d in SWEEP_DISTANCES))
    for i, p in enumerate(ps):
        print(f"  {p:.4f}   " + "".join(f"{res['p_L'][d][i]:<11.4f}" for d in SWEEP_DISTANCES))
    if np.isfinite(th):
        print(f"\n  threshold ~ {th:.4f} +/- {sd:.4f}   "
              f"(crossings: { {k: round(v, 4) for k, v in pairs.items()} })")
    else:
        print("\n  No crossing found in this range (try more SHOTS or a different range).")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}threshold_{model['key']}.png"
    plot_threshold(res, th, sd, path, model["name"], pairs)
    print(f"  plot saved to {path}")

def run_comparison(model, d, p):
    kind = model["profile"]
    info = PROFILE_PARAMS[kind]
    print(f"\nDecoder comparison: d={d}, mean p={p}, strengths {info['sweep']}, "
          f"{DEVICE_SAMPLES} random chips each ...")
    res = decoder_comparison(d, p, kind, info["sweep"])
    print("\n  strength      knows rates    assumes uniform")
    for prm, a, b in zip(info["sweep"], res["informed"], res["mismatched"]):
        print(f"  {str(prm):<13} {a:<14.5f} {b:.5f}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/decoder_comparison_{kind}.png"
    plot_decoder_comparison(res, path, info["xlabel"])
    print(f"  plot saved to {path}")

def run_detection_map(model, d, p, param):
    print(f"\nDetection map: d={d}, p={p}, {SHOTS:,} shots ...")
    dm = detection_map(d, p, model["profile"], param)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/detection_map_{model['key']}.png"
    plot_detection_map(dm, path)
    t, u = agg(dm["det_true"]), agg(dm["det_uniform"])
    print(f"  busiest detector fires {max(t.values()):.3f} vs {max(u.values()):.3f} on a uniform chip")
    print(f"  plot saved to {path}")

def menu():
    print("Surface code simulator")
    print(f"Code size d = {DISTANCE}, up to {SHOTS:,} shots per point "
          f"(edit the SETTINGS block at the top to change)")
    while True:
        i = ask_choice("Choose a noise model:",
                       [(m["name"], m["desc"]) for m in NOISE_MODELS])
        model = NOISE_MODELS[i]
        uneven = model["profile"] != "uniform"
        actions = [("Single run", f"Logical error rate at d={DISTANCE} for one error rate p."),
                   ("Threshold sweep", f"Compare sizes d={SWEEP_DISTANCES} over a range of p; "
                                       "find where bigger codes start to help. Saves a plot.")]
        if uneven:
            actions += [("Decoder comparison", "Does knowing which qubits are bad help the "
                                               "decoder? Sweeps how uneven the chip is. Saves a plot."),
                        ("Detection map", "Shows where on the chip errors get detected. Saves a plot.")]
        a = ask_choice("What to run:", actions)
        if a in (0, 2, 3):
            p = ask_number("Physical error rate p", model["p"])
        param = ask_profile_param(model["profile"]) if uneven and a in (0, 1, 3) else None
        if a == 0:
            run_single(model, DISTANCE, p, param)
        elif a == 1:
            run_threshold(model, param)
        elif a == 2:
            run_comparison(model, DISTANCE, p)
        else:
            run_detection_map(model, DISTANCE, p, param)
        if input("\nRun something else? [Y/n]: ").strip().lower() in ("n", "no", "q"):
            break

#Batch run of everything (--all) and self-test (--selftest):
def run_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)
    summary = {}
    for model in NOISE_MODELS[:3]:
        _, _, ps = sweep_ps(model)
        log(f"threshold sweep: {model['name']}")
        res = threshold_sweep(SWEEP_DISTANCES, ps, model["level"],
                              shots=SWEEP_SHOTS, fails=SWEEP_FAILURES)
        th, sd, pairs = estimate_threshold(res["ps"], res["p_L"], res["shots"])
        log(f"   p_th = {th:.4f} +/- {sd:.4f}")
        plot_threshold(res, th, sd, f"{OUTPUT_DIR}/threshold_{model['key']}.png", model["name"], pairs)
        summary[model["key"]] = {"p_th": th, "p_th_err": sd,
                                 "pairs": {str(k): v for k, v in pairs.items()},
                                 "ps": ps, "p_L": {str(d): v for d, v in res["p_L"].items()},
                                 "shots": {str(d): v for d, v in res["shots"].items()}}
        if model["key"] == "circuit":
            # far below threshold p_L is tiny, so these points need many more shots
            res2 = threshold_sweep(SWEEP_DISTANCES, list(np.geomspace(0.0007, 0.012, 12)), "circuit",
                                   shots=2_000_000, fails=1_000)
            th2, _, _ = estimate_threshold(res2["ps"], res2["p_L"], res2["shots"])
            th_use = th if np.isfinite(th) else th2
            plot_logical_vs_physical(res2, th_use, f"{OUTPUT_DIR}/logical_vs_physical.png")
    for kind in ("lognormal", "hotspot"):
        log(f"decoder comparison: {kind}")
        info = PROFILE_PARAMS[kind]
        r = decoder_comparison(DISTANCE, 0.004, kind, info["sweep"])
        plot_decoder_comparison(r, f"{OUTPUT_DIR}/decoder_comparison_{kind}.png", info["xlabel"])
        summary["decoders_" + kind] = {"params": [list(p) if isinstance(p, tuple) else p for p in info["sweep"]],
                                       "informed": r["informed"].tolist(),
                                       "mismatched": r["mismatched"].tolist()}
    log("detection map")
    plot_detection_map(detection_map(DISTANCE, 0.003, "hotspot", (3, 15.0)),
                       f"{OUTPUT_DIR}/detection_map_hotspot.png")
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"done; results in {OUTPUT_DIR}/")

def selftest():
    d, p = 5, 0.003
    ref = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                 after_clifford_depolarization=p, before_round_data_depolarization=p,
                                 before_measure_flip_probability=p, after_reset_flip_probability=p)
    base = build_circuit(d, p, "circuit")
    mine = apply_profile(base, make_profile(base, "uniform"))
    assert ref.detector_error_model(decompose_errors=True).approx_equals(
        mine.detector_error_model(decompose_errors=True), atol=1e-9)
    print("OK  uniform profile reproduces Stim's built-in error model exactly")

    prof = make_profile(base, "hotspot", (3, 8.0), seed=1, normalize=False)
    hot = apply_profile(base, prof)
    assert hot.num_detectors == base.num_detectors and hot.num_observables == 1
    rates = qubit_error_rates(hot)
    assert prof.m(max(rates, key=rates.get)) == 8.0
    print("OK  rewritten circuit keeps detectors/observables; hot qubits get 8x rates")

    lo = run_memory(build_circuit(3, 0.001), max_shots=50_000, max_failures=10**9, seed=1).p_L
    hi = run_memory(build_circuit(3, 0.02), max_shots=20_000, max_failures=10**9, seed=1).p_L
    assert lo < 0.01 < hi, (lo, hi)
    print(f"OK  decoder sane: d=3 p_L(p=0.001)={lo:.4f}, p_L(p=0.02)={hi:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Surface-code memory simulator")
    ap.add_argument("--all", action="store_true", help="run every experiment non-interactively")
    ap.add_argument("--selftest", action="store_true", help="run correctness checks")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.all:
        run_all()
    else:
        try:
            menu()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")