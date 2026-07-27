

import numpy as np
import qutip as qt

def create_qubit_state(theta: float, phi: float) -> qt.Qobj:
    """
    Create a normalized single-qubit state from Bloch-sphere angles.

    |psi> = cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>
    """
    if not 0 <= theta <= np.pi:
        raise ValueError("theta must be between 0 and pi.")

    ket_zero = qt.basis(2, 0)
    ket_one = qt.basis(2, 1)

    psi = (
        np.cos(theta / 2) * ket_zero
        + np.exp(1j * phi) * np.sin(theta / 2) * ket_one
    )

    return psi