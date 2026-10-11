# Surface-code quantum memory with Stim: threshold, decoding, non-uniform noise

| `circuits.py`   | builds rotated-surface-code memory circuits at three noise levels (code-capacity, phenomenological, circuit-level) from `stim.Circuit.generated` |
| `noise.py`      | **edits Stim circuits**: rewrites every noise channel so each qubit has its own error rate; hardware profiles (log-normal spread, hot qubits, gradient) |
| `decoding.py`   | sampling + DEM extraction + PyMatching decoding; informed vs mismatched decoder; per-round conversion |
| `analysis.py`   | threshold from curve crossings; sub-threshold exponent fits |
| `experiments.py`| threshold sweep, heterogeneity sweep, detection map |
| `plotting.py`   | figures |
| `run_all.py`    | runs everything -> `results/*.png`, `results/summary.json` |
| `validate.py`   | unit checks |

## Method
1. **Circuit.** `surface_code:rotated_memory_z`: d x d data qubits, d^2-1 ancillas,
   d rounds of stabiliser measurement, logical Z preserved and read out at the end.
2. **Noise.** One reference rate p. Circuit level = DEPOLARIZE2(p) after each CX,
   DEPOLARIZE1(p) after H and on data qubits each round, X_ERROR(p) after resets and
   before measurements. Code-capacity keeps only data noise; phenomenological adds measurement flips.
3. **Per-qubit rates (the Stim "edit").** `apply_profile` walks the circuit (recursing into
   REPEAT blocks) and splits each noise instruction into one instruction per target with
   `p_q = min(p * m_q, cap)`; two-qubit gates use the mean of the two multipliers. Detectors,
   observables and the measurement record are untouched. `validate.py` proves that
   all-ones multipliers reproduce Stim's own detector error model exactly.
4. **Decoder uses the statistics.** `detector_error_model()` turns the noisy circuit into a
   matching graph; PyMatching weights each edge ln((1-p_e)/p_e). So a decoder built from the
   non-uniform circuit "knows" which qubits are bad. A *mismatched* decoder is built from the
   uniform circuit at the same mean p.
5. **Failure rate.** Sample until 1000 failures (or max shots); logical failure = decoder's
   predicted observable flip != true flip. Per-round rate: p_round = (1 - (1-2P)^(1/r))/2.
6. **Threshold.** Crossings of total failure probability (over d rounds) for neighbouring
   distances, linearly interpolated in log-log; reported as mean over pairs. (Using per-round
   rates for the crossing biases the answer upward when rounds = d, an issue found and fixed
   during development.)

## Results (default run, MWPM)
| noise model | threshold |
|---|---|
| circuit-level depolarising (this Stim model) | **0.71% +/- 0.04%** (pairs 3/5: 0.65%, 5/7: 0.73%, 7/9: 0.74%) |
| phenomenological | **4.1% +/- 0.1%** |
| code capacity (depolarising, perfect syndrome) | **14.0% +/- 0.6%** |

Sub-threshold scaling p_L ~ p^k: fitted k = 1.94, 3.01, 3.93, 4.84 for d = 3, 5, 7, 9
vs theory (d+1)/2 = 2, 3, 4, 5.

Non-uniform noise (d = 5, mean p = 0.004, total failure over 5 rounds):
- Log-normal spread sigma 0 -> 1.5: informed decoder 0.0079 -> 0.0037; mismatched 0.0079 -> 0.0057.
- 3 hot qubits at 10x/20x/40x: informed 0.0056/0.0041/0.0028; mismatched 0.0067/0.0060/0.0052.
- Because the *mean* rate is held fixed, concentrating noise on few qubits makes the rest of the chip
  cleaner, and the informed decoder profits from it; the uniform-assuming decoder only partly does.
  Error bars (std over device realisations) are large; the ordering informed < mismatched is consistent.
- `detection_map.png`: syndrome detectors next to hot qubits fire up to ~2x more often than on a uniform device.

## Caveats
- Thresholds depend on the noise model; the circuit-level number is for Stim's generated circuit
  (no idle noise during CX layers), not a universal value. Finite-size drift of crossings is visible.
- Per-qubit multipliers are an illustrative spec. Edit the profile constructors in `noise.py`, or
  hand `apply_profile` your own `QubitNoiseProfile` (e.g. measured T1/gate fidelities).
- Only independent (uncorrelated) noise, MWPM decoding, memory (no logical gates).

## References
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021). https://arxiv.org/abs/2103.02202
- O. Higgott, C. Gidney, *Sparse Blossom*, Quantum (2025). https://arxiv.org/abs/2303.15933
- O. Higgott, *PyMatching*. https://arxiv.org/abs/2105.13082 ; docs https://pymatching.readthedocs.io
- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. 43, 4452 (2002). https://arxiv.org/abs/quant-ph/0110143
- A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland, *Surface codes: Towards practical large-scale quantum computation*, Phys. Rev. A 86, 032324 (2012). https://arxiv.org/abs/1208.0928
- N. H. Nickerson, B. J. Brown, *Analysing correlated noise on the surface code using adaptive decoding algorithms*, Quantum 3, 131 (2019). https://arxiv.org/abs/1712.00502
- Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature 614, 676 (2023). https://arxiv.org/abs/2207.06431
- Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature (2024/25). https://arxiv.org/abs/2408.13687
- Stim docs / getting started: https://github.com/quantumlib/Stim
