"""
circuits.py -- build rotated-surface-code memory circuits with Stim.

Three noise "levels" (standard hierarchy in the QEC literature):

  code_capacity     : only data qubits suffer depolarising noise; syndrome
                      extraction is perfect.  (Dennis et al. 2002; Fowler 2012)
  phenomenological  : data-qubit noise + measurement bit-flips.
  circuit           : depolarising noise after every gate, reset flips,
                      measurement flips and data-qubit noise each round
                      (the "SD6"-style model used in Stim's own tutorials).

All of them are produced by `stim.Circuit.generated` with one reference rate p,
and then `noise.apply_profile` rescales rates per qubit.
"""
from __future__ import annotations

from typing import Optional

import stim

from noise import QubitNoiseProfile, apply_profile

LEVELS = ("code_capacity", "phenomenological", "circuit")


def build_circuit(distance: int, p: float, level: str = "circuit",
                  rounds: Optional[int] = None, basis: str = "z",
                  profile_factory=None) -> stim.Circuit:
    """Rotated surface-code memory experiment."""
    
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if rounds is None:
        rounds = 1 if level == "code_capacity" else distance
    kw = {}
    if level == "code_capacity":
        kw = dict(before_round_data_depolarization=p)
    elif level == "phenomenological":
        kw = dict(before_round_data_depolarization=p, before_measure_flip_probability=p)
    else:
        kw = dict(after_clifford_depolarization=p,
                  before_round_data_depolarization=p,
                  before_measure_flip_probability=p,
                  after_reset_flip_probability=p)

        
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        rounds=rounds, distance = distance, 
        **kw)
    if profile_factory is not None:
        circuit = apply_profile(circuit, profile_factory(circuit))
    return circuit
