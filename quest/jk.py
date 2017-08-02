"""
Contains the JK computers (or their bindings)
"""
from . import core

import numpy as np
import psi4


def build_JK(mints, jk_type, auxiliary=None):
    """
    Construct a JK object of various types with the given specifications.

    Parameters
    ----------
    mints : psi4.core.MintsHelper
        Psi4 BasisSet object
    jk_type : str (PK,)
        The type of JK object to construct
    auxiliary : psi4.core.BasisSet (None)
        The auxiliary basis for DF computations

    Returns
    -------
    jk_object : JK
        A initialized JK object

    Notes
    -----
    For DF the basis is automatically constructed as the complementary JK object.

    Examples
    --------

    jk = build_JK(mints, "PK")
    J, K = jk.compute_JK(C_left)
    ...

    """

    if jk_type == "PK":
        return PKJK(mints)
    else:
        raise KeyError("build_JK: Unknown JK type '%s'" % jk_type)


class PKJK(object):
    """
    Constructs a "PK" JK object. This effectively holds two supermatrices which the inner product is
    then taken for speed.

    J_pq[D_rs] = I_prqs D_rs
    K_pq[D_rs] = I_pqrs D_rs

    """

    def __init__(self, mints):
        """
        Initialized the JK object from a MintsHelper object.
        """
        self.nbf = mints.nbf()
        self.I = np.asarray(mints.ao_eri())

    def compute_JK(self, C_left, C_right=None):
        """
        Compute the J and K matrices for Cocc orbitals
        """

        if C_right is None:
            C_right = C_left

        D = np.dot(C_right, C_left.T)

        J = np.zeros((self.nbf, self.nbf))
        K = np.zeros((self.nbf, self.nbf))
        core.compute_PKJK(self.I, D, J, K)

        return J, K
