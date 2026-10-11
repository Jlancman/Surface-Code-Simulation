"""
Hardware-heterogeneous noise for Stim circuits.


  1. let Stim generate the circuit with a single reference probability p;
  2. walk the circuit and *rewrite every noise instruction*
        X_ERROR / Z_ERROR / DEPOLARIZE1 / DEPOLARIZE2
     into one instruction per target (or per target pair), with the probability
        p_target = clamp( p * multiplier(qubit) ).
  
A `QubitNoiseProfile` is just a dictionary  qubit_index -> multiplier  (1.0 = nominal).
A 2-qubit gate on (a, b) uses the mean of the two multipliers.  Because the
noise instructions are rewritten *in place*, detectors, observables and the
measurement record are untouched, so Stim's detector-error-model (DEM)
extraction and sampling work exactly as for the uniform circuit.

Ref: Gidney, "Stim: a fast stabilizer circuit simulator", Quantum 5, 497 (2021),
     https://arxiv.org/abs/2103.02202
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import stim

# Stim noise channels whose probability argument we rescale.
_SINGLE_QUBIT_NOISE = {"X_ERROR", "Z_ERROR", "Y_ERROR", "DEPOLARIZE1"}
_TWO_QUBIT_NOISE = {"DEPOLARIZE2"}
_MAX_P = {"X_ERROR": 0.5, "Z_ERROR": 0.5, "Y_ERROR": 0.5,
          "DEPOLARIZE1": 0.75, "DEPOLARIZE2": 15 / 16}


@dataclass
class QubitNoiseProfile:
    """Per-qubit error-rate multipliers plus a human-readable name."""
    name: str
    multipliers: Dict[int, float]          # qubit index -> multiplier
    coords: Dict[int, Tuple[float, float]]  # qubit index -> (x, y) lattice coords

    def m(self, q: int) -> float:
        return self.multipliers.get(q, 1.0)

    def mean_multiplier(self) -> float:
        return float(np.mean(list(self.multipliers.values())))


# ----------------------------------------------------------------------------
# Profile constructors (the "higher level specification" -- swap in your own)
# ----------------------------------------------------------------------------
def _coords(circuit: stim.Circuit) -> Dict[int, Tuple[float, float]]:
    return {q: (c[0], c[1]) for q, c in circuit.get_final_qubit_coordinates().items()}


def uniform_profile(circuit: stim.Circuit) -> QubitNoiseProfile:
    co = _coords(circuit)
    return QubitNoiseProfile("uniform", {q: 1.0 for q in co}, co)


def lognormal_profile(circuit: stim.Circuit, sigma: float, seed: int = 0,
                      normalize: bool = True) -> QubitNoiseProfile:
    """Multipliers ~ LogNormal(0, sigma).  Fabrication spread of error rates is
    commonly heavy-tailed/log-normal-like.  With normalize=True the *mean*
    multiplier is forced to 1, so the average physical error rate is unchanged
    and only the *spread* varies (a fair comparison against uniform noise)."""
    co = _coords(circuit)
    rng = np.random.default_rng(seed)
    qs = sorted(co)
    s = rng.lognormal(0.0, sigma, size=len(qs)) if sigma > 0 else np.ones(len(qs))
    if normalize:
        s = s / s.mean()
    return QubitNoiseProfile(f"lognormal(sigma={sigma})", dict(zip(qs, s)), co)


def hotspot_profile(circuit: stim.Circuit, n_bad: int, factor: float,
                    seed: int = 0, data_only: bool = False,
                    normalize: bool = True) -> QubitNoiseProfile:
    """`n_bad` randomly chosen qubits are `factor` times noisier than the rest."""
    co = _coords(circuit)
    rng = np.random.default_rng(seed)
    pool = sorted(co)
    if data_only:
        # data qubits sit on odd/odd coordinates in Stim's rotated layout
        pool = [q for q in pool if co[q][0] % 2 == 1 and co[q][1] % 2 == 1]
    bad = set(rng.choice(pool, size=min(n_bad, len(pool)), replace=False).tolist())
    m = {q: (factor if q in bad else 1.0) for q in co}
    if normalize:
        mean = np.mean(list(m.values()))
        m = {q: v / mean for q, v in m.items()}
    return QubitNoiseProfile(f"hotspot(n={n_bad}, x{factor})", m, co)


def gradient_profile(circuit: stim.Circuit, ratio: float,
                     normalize: bool = True) -> QubitNoiseProfile:
    """Error rate increases linearly across the chip (left edge nominal,
    right edge `ratio` times worse) -- e.g. a thermal / wiring gradient."""
    co = _coords(circuit)
    xs = np.array([co[q][0] for q in sorted(co)])
    t = (xs - xs.min()) / max(xs.max() - xs.min(), 1e-9)
    s = 1 + (ratio - 1) * t
    if normalize:
        s = s / s.mean()
    return QubitNoiseProfile(f"gradient(x{ratio})", dict(zip(sorted(co), s)), co)


# ----------------------------------------------------------------------------
# The circuit editor
# ----------------------------------------------------------------------------
def apply_profile(circuit: stim.Circuit, profile: QubitNoiseProfile) -> stim.Circuit:
    """Return a copy of `circuit` whose noise channels are rescaled per qubit."""
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                inst.repeat_count, apply_profile(inst.body_copy(), profile)))
            continue
        name = inst.name
        args = inst.gate_args_copy()
        if name in _SINGLE_QUBIT_NOISE and args:
            p0 = args[0]
            for t in inst.targets_copy():
                p = min(p0 * profile.m(t.value), _MAX_P[name])
                out.append(name, [t], [p])
        elif name in _TWO_QUBIT_NOISE and args:
            p0 = args[0]
            ts = inst.targets_copy()
            for a, b in zip(ts[0::2], ts[1::2]):
                m = 0.5 * (profile.m(a.value) + profile.m(b.value))
                out.append(name, [a, b], [min(p0 * m, _MAX_P[name])])
        else:
            out.append(inst)
    return out


def qubit_error_rates(circuit: stim.Circuit) -> Dict[int, float]:
    """Summed first-order probability of any error attached to each qubit
    (diagnostic only; used to colour the lattice map)."""
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
