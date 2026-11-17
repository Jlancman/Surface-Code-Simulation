from dataclasses import dataclass
import numpy as np
from typing import Optional
import stim
import pymatching

#Applies for running a single round, as opposed to a threshold sweep
SHOTS = 20_000                # max shots per data point (raise to ~400_000 for publication-quality curves)
MAX_FAILURES = 300            # stop a data point early once this many logical failures are seen

#Sampling and decoding:
@dataclass
class Result:
    shots: int
    failures: int

    @property
    def p_L(self) -> float:
        return self.failures / self.shots if self.shots else float("nan")
    #logical error rate

    @property
    def stderr(self) -> float:
        p = self.p_L
        return float(np.sqrt(max(p * (1 - p), 1e-12) / self.shots)) if self.shots else float("nan")
    #binomial standard error

def run_memory(sample_circuit: stim.Circuit, decoder_circuit: Optional[stim.Circuit] = None,
               max_shots: int = SHOTS, max_failures: int = MAX_FAILURES,
               batch: int = 10_000, seed: Optional[int] = None) -> Result:
    """Sample until `max_failures` failures or `max_shots` shots.
    The decoder's matching graph comes from `decoder_circuit` (default: the
    sampling circuit itself, i.e. a decoder that knows the true noise)."""
    dem = (decoder_circuit or sample_circuit).detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    #goes through all noise lines in the circuit to determine which errors would be set off,
    #and whether there would be a logical error. Decomposes errors and approximates mutually 
    #exclusive Pauli channel errors
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = sample_circuit.compile_detector_sampler(seed=seed)
    shots = fails = 0
    while shots < max_shots and fails < max_failures:
        n = min(batch, max_shots - shots)
        det, obs = sampler.sample(n, separate_observables=True)
        #samples which detectors were triggered per shot
        pred = matcher.decode_batch(det)
        fails += int(np.count_nonzero(np.any(pred != obs, axis=1)))
        #counts failures based on sampled detectors
        shots += n
    return Result(shots, fails)

def per_round(p_L: float, rounds: int) -> float:
    """Total failure over `rounds` rounds -> per-round rate: 1-2P = (1-2p_r)^rounds."""
    return 0.5 * (1 - (1 - 2 * min(p_L, 0.5)) ** (1.0 / rounds))
    #allows for comparison between runs with different numbers of rounds

def detection_rates(circuit: stim.Circuit, shots: int, seed: int = 1):
    """Mean firing probability of every detector -> {(x, y, t): prob}."""
    det = circuit.compile_detector_sampler(seed=seed).sample(shots)
    coords = circuit.get_detector_coordinates()
    mean = det.mean(axis=0)
    return {tuple(coords[i]): float(mean[i]) for i in range(len(mean))}
#gives information about the detectors
#returns the position of each detector