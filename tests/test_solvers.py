"""
This file tests the conjugate solver function in the quest.
"""

from quest import solvers
import pytest
import numpy as np
import scipy
import scipy.sparse.linalg

# Set a seed so it is not truly random
np.random.seed(0)


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
    M = np.diag(A)
    x1 = solvers.helper_PCG_direct(A, b, tol, max_iteration, x0, M)

    # Check our solution vs. numpy
    print(" Solution matches with numpy's cg solver: %s" % np.allclose(x, x1, 1e-9))
    assert np.allclose(x, x1, 1e-9)


def test_helper_PCG():
    # Generate sample symmetric positive-definite matrix A and RHS vector b
    n = 100
    A = np.random.normal(size=[n, n])
    A = np.dot(A.T, A)

    def hess_A(x):
        return np.dot(A, x)

    b = np.ones(n)
    max_iteration = 2 * n

    # Numpy's linear equation solver
    x = np.linalg.solve(A, b)

    # Our implementation
    tol = 1e-14
    x0 = np.zeros(n)
    # M = np.diag(A)
    M = np.ones((A.shape[0]))
    x1 = solvers.helper_PCG(hess_A, b, tol, max_iteration, x0, M)

    # Check our solution vs. numpy
    print(" Solution matches with numpy's cg solver: %s" % np.allclose(x, x1, 1e-9))
    assert np.allclose(x, x1, 1e-9)
