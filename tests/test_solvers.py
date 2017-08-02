"""
This file tests the conjugate solver function in the quest.
"""

import quest
import pytest
import numpy as np

def test_helper_PCG_direct():
# Generate sample symmetric positive-definite matrix A and RHS vector b
    n = 100
    A = np.random.normal(size=[n, n])
    A = np.dot(A.T, A)
    b = np.ones(n)
    max_iteration = 2 * n
    
    # Numpy's linear equation solver
    x = np.linalg.solve(A, b)
    
    # Our implementation
    tol = 1e-14
    x0 = np.zeros(n)
    M = np.ones_like(A)
    x1 = quest.solvers.helper_PCG_direct(A, b, tol, max_iteration, x0, M)
    
    # Check our solution vs. numpy
    print(" Solution matches with numpy's cg solver: %s" % np.allclose(x, x1, 1e-9))
    assert np.allclose(x, x1, 1e-9)
