import stim
from typing import Optional, Tuple
from noise import make_profile, apply_profile, MAX_P
"""
Creates a surface code circuits, and applies an uneven noise
profile to it using noise.py
"""

#Circuits:
LEVELS = ("code_capacity", "phenomenological", "circuit", "repetition")
ROUNDS_PER_D = 3 #STIM's getting started uses a 3*d; can default to d rounds as well
PROBS = [None, None, None, None]

#how many rounds of error checking per level of noise
def rounds_for(level: str, distance: int) -> int:
    return 1 if level == "code_capacity" else ROUNDS_PER_D*distance

def build_circuit(distance: int, p: float, level: str = "circuit",
                  profile: Optional[Tuple] = None, seed: int = 0,
                  data_only: bool = False) -> stim.Circuit:
    """Rotated surface-code Z-memory experiment.

    profile = (kind, param) to make qubit error rates non-uniform, or None.
    """
    probs = [p if q is None else q for q in PROBS]   # local copy: PROBS itself never changes
    dep1 = min(probs[0], MAX_P["DEPOLARIZE1"])
    xmeasure = min(probs[1], MAX_P["X_ERROR"])
    xnot = min(probs[2], MAX_P["DEPOLARIZE2"])
    xreset = min(probs[3], MAX_P["X_ERROR"])

    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if level == "code_capacity":
        kw = dict(before_round_data_depolarization = dep1)
        #only adds a depolarizing gate with prob p on data qubits before each round
    elif level == "phenomenological":
        kw = dict(before_round_data_depolarization = dep1, 
                  before_measure_flip_probability = xmeasure)
        #same as above; also adds an x error with prob p before each measurement
    elif level == "repetition":
        kw = dict(before_round_data_depolarization = dep1)
    else:
        kw = dict(after_clifford_depolarization = xnot,
                  before_round_data_depolarization = dep1,
                  before_measure_flip_probability = xmeasure,
                    after_reset_flip_probability = xreset)
        #same as above, also has depolarizing after CNOT and x error after resets, all with prob p
    task = "repetition_code:memory" if level == "repetition" else "surface_code:rotated_memory_z"
    circuit = stim.Circuit.generated(task,
                                     rounds=rounds_for(level, distance),
                                     distance=distance, **kw) #to properly compare with riverlane's code, use rotated_memory_x
    #use inbuilt stim functions to write a surface code with the given noise
    #using these inbuilt functions makes it so that each form of gate error
    #has the same probability of error across the code, as opposed to qubits, 
    #who have varying error per-qubit. IE: every measurement has the same error,
    #every reset has the same error, etc.
    if profile is not None and profile[0] != "uniform":
        prof = make_profile(circuit, profile[0], profile[1], seed=seed, data_only=data_only)
        circuit = apply_profile(circuit, prof)
        #call to noise.py to make and apply the noise profile on the qubits
    return circuit