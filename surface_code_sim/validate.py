"""validate.py -- sanity checks that our Stim editing is correct."""
import stim
import numpy as np
import noise as nz
from circuits import build_circuit
from decoding import run_memory

d, p = 5, 0.003
ref = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                             after_clifford_depolarization=p, before_round_data_depolarization=p,
                             before_measure_flip_probability=p, after_reset_flip_probability=p)
mine = build_circuit(d, p, "circuit", profile_factory=nz.uniform_profile)
dem_a = ref.detector_error_model(decompose_errors=True)
dem_b = mine.detector_error_model(decompose_errors=True)
assert dem_a.approx_equals(dem_b, atol=1e-9), "uniform profile must reproduce Stim's DEM"
print("OK  uniform profile reproduces stim's built-in DEM exactly")

# a hot qubit must raise the probability of the detection events next to it
base = build_circuit(d, p, "circuit")
prof = nz.hotspot_profile(base, 3, 8.0, seed=1, normalize=False)
hot = nz.apply_profile(base, prof)
assert hot.num_detectors == base.num_detectors and hot.num_observables == 1
rates = nz.qubit_error_rates(hot)
worst = max(rates, key=rates.get)
assert prof.m(worst) == 8.0
print("OK  rewritten circuit keeps detectors/observables; hot qubits get 8x rates")

# known-good number: d=3 p=0.001 should fail rarely, p=0.02 should be far worse
lo = run_memory(build_circuit(3, 0.001), max_shots=50_000, max_failures=10**9, seed=1).p_L
hi = run_memory(build_circuit(3, 0.02), max_shots=20_000, max_failures=10**9, seed=1).p_L
assert lo < 0.01 < hi, (lo, hi)
print(f"OK  decoder sane: d=3 p_L(p=0.001)={lo:.4f}, p_L(p=0.02)={hi:.3f}")
