import numpy as np
import stim
from dataclasses import dataclass
from typing import Dict, Tuple

# Noise models shown in the menu
# level   : which Stim noise channels are switched on
# profile : how error rates vary between qubits (circuit level only)
# Describes how much worse a qubit is using a multiplier
# Converts a circuit into an identical circuits with per-qubit error rates rescaled by the multiplier
# Gates and measurements face the same probability of error

NOISE_MODELS = [
    dict(key="code_capacity", level="code_capacity", profile="uniform",
         name="Code capacity",
         desc="Only data qubits get errors (random X/Y/Z flips); "
              "checking for errors is perfect; only 1 round of error correction is done.",
         p=0.10, sweep=(0.08, 0.20)),
    dict(key="phenomenological", level="phenomenological", profile="uniform",
         name="Phenomenological",
         desc="Data-qubit errors, plus each syndrome measurement can "
              "give the wrong answer. Models faulty error checking.",
         p=0.03, sweep=(0.02, 0.05)),
    dict(key="circuit", level="circuit", profile="uniform",
         name="Circuit level",
         desc="Every gate, reset and measurement can fail, all qubits equally. "
              "Closest to real hardware.",
         p=0.004, sweep=(0.006, 0.016)),
    dict(key="lognormal", level="circuit", profile="lognormal",
         name="Circuit level, uneven qubits (random spread)",
         desc="Circuit-level noise where each qubit's error rate is randomly "
              "better or worse than average.",
         p=0.004, sweep=(0.004, 0.012)),
    dict(key="hotspot", level="circuit", profile="hotspot",
         name="Circuit level, a few bad qubits",
         desc="Circuit-level noise where a handful of qubits are much noisier "
              "than the rest (average error rate kept the same).",
         p=0.004, sweep=(0.004, 0.012)),
    dict(key="repetition", level="repetition", profile="uniform",
         name="Repetition code (Stim turorial)", 
         desc = "A 1D chain that only protects against bit flips. Data qubits get" \
         "random X/Y/Z flips each round; checking is perfect.", 
         p = 0.1, sweep = (0.05, 0.5))
]
# dictionary of noise models, each with a key, 
# level, profile, name, description, default error rate p,
# and sweep range for p.

# Default strength of each non-uniform profile, and how the menu asks for it.
PROFILE_PARAMS = {
    "uniform":   dict(default=None),
    "lognormal": dict(default=1.0, prompt="Spread sigma (0 = uniform, 1.5 = very uneven)",
                      sweep=[0.0, 0.5, 1.0, 1.5], xlabel="log-normal spread $\\sigma$ of qubit error rates"),
    "hotspot":   dict(default=(3, 20.0), prompt="How many times worse the bad qubits are",
                      sweep=[(0, 1.0), (3, 10.0), (3, 20.0), (3, 40.0)],
                      xlabel="error-rate factor of hot qubits (x nominal)"),
}

# Per-qubit noise: profiles and the Stim circuit editor
_SINGLE_QUBIT_NOISE = {"X_ERROR", "Z_ERROR", "Y_ERROR", "DEPOLARIZE1"}
_TWO_QUBIT_NOISE = {"DEPOLARIZE2"}
MAX_P = {"X_ERROR": 0.5, "Z_ERROR": 0.5, "Y_ERROR": 0.5,
          "DEPOLARIZE1": 0.75, "DEPOLARIZE2": 15 / 16}
#Maximum error rates for each noise channel, used to prevent probabilities from exceeding physical limits

@dataclass
class QubitNoiseProfile:
    """Per-qubit error-rate multipliers (1.0 = nominal) plus lattice coords."""
    name: str
    multipliers: Dict[int, float] #stores multipliers for all qubits
    coords: Dict[int, Tuple[float, float]] #qubits coords on surface

    def m(self, q: int) -> float: #call multiplier for qubit q
        return self.multipliers.get(q, 1.0)

#returns a dictionary mapping qubit indices to their coordinates in the circuit
def _coords(circuit: stim.Circuit) -> Dict[int, Tuple[float, float]]:
    return {q: (c[0], c[1]) for q, c in circuit.get_final_qubit_coordinates().items()}

def _normalized(qs, s, normalize):
    s = np.asarray(s, float)
    if normalize:
        s = s / s.mean()   # keep the average error rate fixed at 1.0; only the spread changes
    return dict(zip(qs, s)) # pairs each qubit with its multiplier

# builds a per-qubit multiplier map based on the specified noise model
def make_profile(circuit: stim.Circuit, kind: str, param=None, seed: int = 0,
                 data_only: bool = False, normalize: bool = True) -> QubitNoiseProfile:
    """Build a per-qubit multiplier map using QubitNoiseProfile.

    uniform   : every qubit at 1.0
    lognormal : LogNormal(0, sigma)
    hotspot   : param = (n_bad, factor); n_bad random qubits are `factor` x worse
    """
    co = _coords(circuit) #get qubit coordinates
    qs = sorted(co) #sort qubit indices
    rng = np.random.default_rng(seed)
    if kind == "uniform": #reference: every qubit has the same error rate multiplier of 1.0
        return QubitNoiseProfile("uniform", {q: 1.0 for q in qs}, co)
    if kind == "lognormal": #qubit error rates are drawn from a log-normal distribution
        sigma = float(param)
        s = rng.lognormal(0.0, sigma, size=len(qs)) if sigma > 0 else np.ones(len(qs))
        return QubitNoiseProfile(f"lognormal(sigma={sigma})", _normalized(qs, s, normalize), co)
    if kind == "hotspot": #picks bad qubits and makes them worse by a factor. Av error rate stays the same
        n_bad, factor = int(param[0]), float(param[1])
        pool = qs #selected qubits
        if data_only:   # data qubits sit on odd/odd coordinates in Stim's rotated layout
            pool = [q for q in qs if co[q][0] % 2 == 1 and co[q][1] % 2 == 1]
        bad = set(rng.choice(pool, size=min(n_bad, len(pool)), replace=False).tolist()) if n_bad else set()
        #checks to limit the number of bad qubits
        s = [factor if q in bad else 1.0 for q in qs]
        return QubitNoiseProfile(f"hotspot(n={n_bad}, x{factor})", _normalized(qs, s, normalize), co)
    raise ValueError(f"unknown profile {kind!r}")

def apply_profile(circuit: stim.Circuit, profile: QubitNoiseProfile) -> stim.Circuit:
    """Copy of `circuit` with every noise channel rescaled per qubit.
    Two-qubit noise on (a, b) uses the mean of the two multipliers."""
    out = stim.Circuit()
    for inst in circuit: #cycle through all blocks in the circuit
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                inst.repeat_count, apply_profile(inst.body_copy(), profile)))
            continue #recursive block in case of a repeated section
        name, args = inst.name, inst.gate_args_copy()
        if name in _SINGLE_QUBIT_NOISE and args: #handle single qubit errors
            for t in inst.targets_copy(): #go through target qubits
                out.append(name, [t], [min(args[0] * profile.m(t.value), MAX_P[name])])
                #append additional noise blocks
        elif name in _TWO_QUBIT_NOISE and args: #handle two qubit errors
            ts = inst.targets_copy()
            for a, b in zip(ts[0::2], ts[1::2]):
                m = 0.5 * (profile.m(a.value) + profile.m(b.value))
                #take mean of two qubit gate on a bad qubit and a good one
                out.append(name, [a, b], [min(args[0] * m, MAX_P[name])])
        else:
            out.append(inst) #copy all other gates unchanged
    return out

def qubit_error_rates(circuit: stim.Circuit) -> Dict[int, float]:
    """Summed first-order error probability touching each qubit (diagnostic)."""
    tot: Dict[int, float] = {}

    def walk(c, mult=1):
        for inst in c:
            if isinstance(inst, stim.CircuitRepeatBlock):
                walk(inst.body_copy(), mult * inst.repeat_count)
            elif inst.name in (_SINGLE_QUBIT_NOISE | _TWO_QUBIT_NOISE) and inst.gate_args_copy():
                p = inst.gate_args_copy()[0]
                for t in inst.targets_copy():
                    tot[t.value] = tot.get(t.value, 0.0) + mult * p
    walk(circuit)
    return tot