

import numpy as np 

def hermitian_operator(a, b_real, b_imag, d,): 
    """
    Construct a general 2 × 2 Hermitian matrix.

    H = [[a,              b_real + i*b_imag],
         [b_real-i*b_imag,              d]]

    Parameters
    ----------
    a, d : float
        Real diagonal elements.

    b_real, b_imag : float
        Real and imaginary parts of the upper-right element.

    Returns
    -------
    numpy.ndarray
        A 2 × 2 complex Hermitian matrix.
    """
    return np.array(
        [
            [a, b_real + 1j * b_imag],
            [b_real - 1j * b_imag, d],
        ] )