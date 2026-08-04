

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

def schmidt_decomposition(psi, d_A, d_B, tol=1e-12):
    """
    Compute the Schmidt decomposition of a bipartite pure state.

    The state is decomposed as

        |psi> = sum_k s_k |u_k>_A |v_k>_B

    where s_k are the Schmidt coefficients and |u_k>, |v_k>
    are orthonormal Schmidt basis states.
    np.asarray(): converts an input into a NumPy array
    Parameters
    ----------
    psi : Bipartite pure-state vector with length d_A * d_B.
    d_A : Hilbert-space dimension of subsystem A.
    d_B : Hilbert-space dimension of subsystem B.
    
    tol : float, optional
        Singular values smaller than this tolerance are discarded.

    Returns
    -------
    schmidt_coefficients : The nonzero Schmidt coefficients.
    basis_A : Matrix whose columns are the Schmidt basis vectors of subsystem A.
    basis_B : Matrix whose columns are the Schmidt basis vectors of subsystem B.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1) 
    
    if not isinstance(d_A, int) or not isinstance(d_B, int):
        raise TypeError("d_A and d_B must be integers.")

    if d_A <= 0 or d_B <= 0:
        raise ValueError("d_A and d_B must be positive.")

    expected_size = d_A * d_B

    if psi.size != expected_size:
        raise ValueError(
            f"State-vector length is {psi.size}, but "
            f"d_A * d_B = {expected_size}."
        )

    norm = np.linalg.norm(psi)

    if norm < tol:
        raise ValueError("The state vector cannot be the zero vector.")

    psi = psi / norm  # Normalize the state.
    coefficient_matrix = psi.reshape(d_A, d_B) #Convert the flat state vector into its coefficient matrix.
    U, s, Vh = np.linalg.svd(coefficient_matrix,full_matrices=False,) # coefficient_matrix = U @ diag(s) @ Vh
    keep = s > tol   # Remove numerically zero Schmidt coefficients.
    schmidt_coefficients = s[keep]
    basis_A = U[:, keep]

    basis_B = Vh[keep, :].T # The columns are the subsystem-B Schmidt vectors.
    return schmidt_coefficients, basis_A, basis_B
