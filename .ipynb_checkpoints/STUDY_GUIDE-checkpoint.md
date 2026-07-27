# Computational Quantum Dynamics Study Package

This package contains only the lecture material required for studying Computational Quantum Dynamics.

## Recommended study order

### 00. Preparation and Computational Basics

**Notebook:** `00_Preparation_Basics.ipynb`

**Main topics:**

- Python and scientific-computing preparation
- Computational complexity
- Dense and sparse matrices
- NumPy and SciPy eigenvalue solvers
- LAPACK, ARPACK and BLAS

### 01. Quantum Mechanics Basics

**Notebook:** `01_Lecture_1_Quantum_Mechanics_Basics.ipynb`

**Main topics:**

- Postulates of quantum mechanics
- Quantum states and normalization
- Observables and operators
- Expectation values
- Schrödinger equation

### 02. Numerical Linear Algebra

**Notebook:** `02_Lecture_2_Numerical_Linear_Algebra.ipynb`

**Main topics:**

- Properties of matrices and operators
- Hermitian matrices
- Eigenvalue problems
- QR decomposition and QR iteration
- Sparse matrix formats
- Numerical performance and matrix structure

### 03. Split-Step Fourier Method

**Notebook:** `03_Lecture_3_Split_Step_Fourier_Method.ipynb`

**Main topics:**

- Time-dependent Schrödinger equation
- Split-operator method
- Trotter decomposition
- Strang decomposition
- Fourier transformation
- Fast Fourier transform

### 04. Many-Body Quantum Systems and Approximations

**Notebook:** `04_Lecture_4_Many_Body_Approximations.ipynb`

**Main topics:**

- Many-body Hilbert spaces
- Curse of dimensionality
- Basis-set expansion
- Basis truncation
- Ground and low-energy states
- Approximation methods

### 05. Spin-½ Systems

**Notebook:** `05_Lecture_5_Spin_Half_Systems.ipynb`

**Main topics:**

- Single spin-½ states
- State normalization
- Pauli matrices
- Spin operators
- Tensor-product Hilbert spaces
- Collective spin systems

### 06. ODE Solvers and Quantum Time Evolution

**Notebook:** `06_Lecture_6_ODE_Solvers_and_TDSE.ipynb`

**Main topics:**

- Schrödinger equation as an ODE
- Exact time evolution
- Numerical integration
- Euler and Runge–Kutta methods
- Accuracy and stability
- Time-dependent quantum dynamics

### 07. Collective Spins and Husimi Functions

**Notebook:** `07_Lecture_7_Collective_Spins_and_Husimi_Functions.ipynb`

**Main topics:**

- Bloch-sphere review
- Collective spin operators
- Spin-coherent states
- Quantum-state visualization
- Husimi functions
- Phase-space representations

### 08. Mixed States and Entanglement

**Notebook:** `08_Lecture_8_Mixed_States_and_Entanglement.ipynb`

**Main topics:**

- Pure and mixed quantum states
- Density matrices
- Partial trace
- Reduced density matrices
- Quantum entanglement
- Entanglement measures

### 09. Matrix Product States

**Notebook:** `09_Lecture_9_Matrix_Product_States.ipynb`

**Main topics:**

- Tensor-network notation
- Diagrammatic tensor notation
- Matrix product states
- Entanglement in MPS
- Expectation values and observables
- Many-body state representation

### 10. Variational Optimization of Matrix Product States

**Notebook:** `10_Lecture_10_Variational_MPS_Optimization.ipynb`

**Main topics:**

- Variational principle
- MPS variational ansatz
- Ground-state search
- Local tensor optimization
- Energy minimization
- Convergence of the optimization

### 11. Open Quantum Systems

**Notebook:** `11_Lecture_11_Open_Quantum_Systems.ipynb`

**Main topics:**

- Introduction to open systems
- System and environment
- Density-matrix evolution
- Quantum master equation
- Lindblad equation
- Dissipation and decoherence

### 12. Variational Monte Carlo

**Notebook:** `12_Supplement_Variational_Monte_Carlo.ipynb`

**Main topics:**

- Variational Monte Carlo method
- Jastrow ansatz
- Many-particle correlations
- Monte Carlo sampling
- Local energy
- JAX and Flax neural-network ansatz

## Important Python libraries

The lecture notebooks may use:

```python
import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import qutip as qt
import jax.numpy as jnp
from flax import linen as nn
import Comp_Quant_Dynam as cqd
```

## Study recommendation

1. Read the theoretical Markdown cells first.
2. Identify the important equations and numerical method.
3. Run the example code one section at a time.
4. Explain every plot in your own words.
5. Practise changing the physical and numerical parameters.
6. Review the exercises only after understanding the lecture notebook.
