

import numpy as np

I = np.array([[1, 0],
              [0, 1]], dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)
Z = np.array([[1,  0],
              [0, -1]], dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1,  1],
                               [1, -1]], dtype=complex)
P0 = np.array([[1, 0],
               [0, 0]], dtype=complex)
P1 = np.array([[0, 0],
               [0, 1]], dtype=complex)
S = np.array([[1, 0],
              [0, 1j]], dtype=complex)
T = np.array([[1, 0],
              [0, np.e**(1j * np.pi / 4)]], dtype=complex)

def rotation_gate(theta, n):
    """
    This function implements a unitary rotation of a single qubit
    by an angle `theta` around an axis `n` on the Bloch sphere.

    The rotation generator is constructed as N = n · σ,
    where σ = (X, Y, Z) are the Pauli matrices.

    Parameters
    theta : Rotation angle
    n : Rotation axis 
    """
    nx, ny, nz = n
    N = nx * X + ny * Y + nz * Z
    R = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * N 
    return R
