

import numpy as np

def density_matrix(ket):
    """
    Construct the density matrix
    """
    ket = np.asarray(ket, dtype=complex)
    return np.outer(ket, ket.conj())


def partial_trace(rho_AB, dA, dB):
    """
    Trace out subsystem B from the bipartite density matrix rho_AB.
    """
    rho_AB = np.asarray(rho_AB, dtype=complex)
    expected_shape = (dA * dB, dA * dB)

    if rho_AB.shape != expected_shape:
        raise ValueError(
            f"rho_AB must have shape {expected_shape}, "
            f"but received {rho_AB.shape}."
        )

    # Convert:
    # rho_AB[(i,j), (i',j')]
    #
    # into:
    # rho_AB[i, j, i', j']
    reshaped_rho = rho_AB.reshape(dA, dB, dA, dB)

    # Trace over subsystem B:
    # rho_A[i,i'] = sum_j rho_AB[i,j,i',j]
    rho_A = np.trace(
        reshaped_rho,
        axis1=1,
        axis2=3
    )

    return rho_A


def purity(rho):
    """
    Calculate the purity

        Tr(rho^2)

    A pure state has purity 1.
    A mixed state has purity less than 1.
    """
    rho = np.asarray(rho, dtype=complex)
    result = np.trace(rho @ rho)
    return result

