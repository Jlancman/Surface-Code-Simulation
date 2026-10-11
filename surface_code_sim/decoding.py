"""
Monte-Carlo sampling + minimum-weight perfect-matching decoding.

    `circuit.compile_detector_sampler()` runs Stim's Pauli-frame simulator and
returns detection events (syndrome *changes*) and the true logical flip.    
    `circuit.detector_error_model(decompose_errors=True)` converts the noisy
circuit into a graph: every independent error mechanism becomes an edge
with probability p_e.  PyMatching turns that into edge weights
w_e = ln((1 - p_e) / p_e).   <- the decoder USES the error statistics here.
    PyMatching's MWPM finds the lowest-total-weight set of edges explaining the
detection events and predicts whether the logical observable flipped.   
    A shot is a logical failure if prediction != truth.
    The decoder circuit may differ from the sampling circuit; that lets us study
a *mismatched* decoder that assumes uniform noise on a non-uniform device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymatching
import stim


@dataclass
class Result:
    shots: int
    failures: int

    @property
    def p_L(self) -> float:
        return self.failures / self.shots if self.shots else float("nan")

    @property
    def stderr(self) -> float:
        p = self.p_L
        return float(np.sqrt(max(p * (1 - p), 1e-12) / self.shots)) if self.shots else float("nan")


def make_matcher(decoder_circuit: stim.Circuit) -> pymatching.Matching:
    dem = decoder_circuit.detector_error_model(decompose_errors=True,
                                               approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def run_memory(sample_circuit: stim.Circuit,
               decoder_circuit: Optional[stim.Circuit] = None,
               max_shots: int = 100_000, max_failures: int = 300,
               batch: int = 10_000, seed: Optional[int] = None) -> Result:
    """Sample until `max_failures` logical failures or `max_shots` shots."""
    matcher = make_matcher(decoder_circuit or sample_circuit)
    sampler = sample_circuit.compile_detector_sampler(seed=seed)
    shots = fails = 0
    while shots < max_shots and fails < max_failures:
        n = min(batch, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = matcher.decode_batch(det)
        fails += int(np.count_nonzero(np.any(pred != obs, axis=1)))
        shots += n
    return Result(shots, fails)


def per_round(p_L: float, rounds: int) -> float:
    """Convert total logical failure probability over `rounds` rounds to a
    per-round rate, assuming independent identical rounds:
        1 - 2 P = (1 - 2 p_round)^rounds."""
    p_L = min(p_L, 0.5)
    return 0.5 * (1 - (1 - 2 * p_L) ** (1.0 / rounds))


def detection_rates(sample_circuit: stim.Circuit, shots: int = 20_000, seed: int = 1):
    """Mean firing probability of every detector -> {(x, y, t): prob}."""
    det = sample_circuit.compile_detector_sampler(seed=seed).sample(shots)
    coords = sample_circuit.get_detector_coordinates()
    mean = det.mean(axis=0)
    return {tuple(coords[i]): float(mean[i]) for i in range(len(mean))}
