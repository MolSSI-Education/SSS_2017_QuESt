import numpy as np


"""
This is a file where DIIS and CG would go.
"""
def DIIS_step(state_list, error_list):
    """Routine that returns a guess for a new state vector
    using DIIS

    Parameters
    ----------
    state_list : list of numpy arrays
                 n previous states to form the new state from
    error_list : list of numpy arrays
                 n previous errors corresponding to `state_list`

    Returns
    -------
    new_state : numpy array
                new state array from DIIS update

    Notes
    -----
    Whichever method calls this routine must keep track of its `state_list` and
    `error_list` and update them.

    Examples
    --------
    new_state = solvers.DIIS_step(state_list, error_list)

    """

    # Checks a few possible exceptions
    if len(state_list) != len(error_list):
        raise Exception("State and error list sizes don't match")

    if len(state_list) < 2:
        raise Exception("Must have at least two previous state vectors for DIIS")

    # Build the B matrix
    B = -1 * np.ones((len(error_list) + 1, len(error_list) + 1))
    B[-1, -1] = 0
    for i in range(len(error_list)):
        for j in range(i, len(error_list)):
            ri_dot_rj = np.vdot(error_list[i], error_list[j])
            B[i, j] = B[j, i] = ri_dot_rj

    # Solve the linear equations
    vec = np.zeros((len(error_list) + 1))
    vec[-1] = -1
    coeff = np.linalg.solve(B, vec)
    new_state = np.zeros_like(state_list[-1])

    # Build the new state!
    for i, prev_state in enumerate(state_list):
        new_state += coeff[i] * prev_state

    return new_state




"""
Contains the conjugate gradient helper function
"""
def helper_PCG_direct(A, b, tol=1e-10, max_iter=None, x0=None, M=None):
    """
    Solves a linear system of equations Ax = b using a preconditioned conjugate 
    gradient method within a user-defined tolerance
	

    Parameters
    ----------
    A : LHS Matrix (2-dimensional numpy array)
    b : RHS vector (1-dimensional numpy array)
    tol : User-defined tolerance for residual convergence (Default = 1.e-10)
    max_iter : User-defined maximum number of iterations for convergence (Default = 2 * len(b))
    x0 : User-specified guess vector (Default = zero-array)
    M : Preconditioner Matrix (2-dimensional numpy array)  (Default = Identity Matrix)  

    Returns
    -------
    x : solution vector of Ax = b

    Notes
    -----
    
    Examples
    --------

    x = helper_PCG_direct(A, b)
    ...

    """

    if max_iter is None:
        max_iter = 2*len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    if M is None:
        M = np.ones((A.shape[0]))

    r = b - np.dot(A, x)
    z = r / M
    n = len(b)
    p = z
    res_k_norm = np.vdot(r, z)
    print("Starting Conjugate Gradient Iterations...\n")
    for iteration in range(2*n):
        Ap = np.dot(A, p)
        alpha = res_k_norm / np.vdot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = r / M
        res_kplus1_norm = np.vdot(r, z)
        beta = res_kplus1_norm / res_k_norm
        res_k_norm = res_kplus1_norm
        rms = np.sqrt(np.sum(r**2) / len(r))
        print('CG Iteration %3d: RMS = %3.8f' % (iteration, rms))
        if (res_kplus1_norm < tol) or (iteration > max_iter):
            print('\nConverged!!\n')
            break
        p = beta * p + z
    return x





"""
Contains the conjugate gradient helper function
"""
def helper_PCG(hess_vec, b, tol=1e-10, max_iter=None, x0=None, M=None):
    """
    Solves a linear system of equations Ax = b using a preconditioned conjugate gradient method within a user-defined tolerance while avoiding storage of the four-dimensional array A (for SCF Hessians)
	

    Parameters
    ----------
    hess_vec : function pointer to Hessian vector multiplication function 
               (hess vec returns a vector Ax)
    b : RHS vector (1-dimensional numpy array)
    tol : User-defined tolerance for residual convergence (Default = 1.e-10)
    max_iter : User-defined maximum number of iterations for convergence (Default = 2 * len(b))
    x0 : User-specified guess vector (Default = zero-array)
    M : Preconditioner Matrix (2-dimensional numpy array)  (Default = Identity Matrix)  

    Returns
    -------
    x : solution vector of Ax = b

    Notes
    -----
    
    Examples
    --------

    x = helper_PCG(hess_vec, b)
    ...

    """

    if max_iter is None:
        max_iter = 2*len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    if M is None:
        M = np.ones_like(hess_vec.shape[0])

    r = b - hess_vec(x)
    z = r / M
    n = len(b)
    p = z
    res_k_norm = np.vdot(r, z)
    print("Starting Conjugate Gradient Iterations...\n")
    for iteration in range(2*n):
        Ap = hess_vec(p)
        alpha = res_k_norm / np.vdot(Ap, p)
        x += alpha * p
        r -= alpha * Ap
        z = r / M
        res_kplus1_norm = np.vdot(r, z)
        beta = res_kplus1_norm / res_k_norm
        res_k_norm = res_kplus1_norm
        rms = np.sqrt(np.sum(r**2) / len(r))
        print('CG Iteration %3d: RMS = %3.8f' % (iteration, rms))
        if (res_kplus1_norm < tol) or (iteration > max_iter):
            print('\nConverged!!\n')
            break
        p = beta * p + z
    return x
