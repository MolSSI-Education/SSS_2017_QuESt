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
    if jk_type == "DF":
        return DFJK(mints)
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


class DFJK(object):
    """
    Constructs a "DF" JK object. This uses density-fitting to accelerate the construction
    of J and K matrices.

    kai_P = I_Prs D_rs
    J_pq[D_rs] = I_Ppq kai_P

    zeta_Pqs = I_Pqs D_s
    K_pq[D_rs] = I_Ppr zeta_Pqr

    """

    def __init__(self, mints):
        """
        Initialized the JK object from a MintsHelper object and prepare the 
        auxiliary integrals:
        
        Parameters
        ----------
        mints     : psi4.core.MintsHelper
        bas       : basis set object
        mol       : molecule object
        basname   : string of the basis set name
        
        """
        self.nbf = mints.nbf()

        # Build the complementary JKFIT basis for the aug-cc-pVDZ basis (for example)
        bas = mints.basisset() 
        mol = bas.molecule()
        aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=bas.name())

        # The zero basis set
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

        # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
        Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
        Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions

        # Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
        metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
        metric.power(-0.5, 1.e-14)
        metric = np.squeeze(metric) # remove the 1-dimensions

        Pls = np.einsum('pq,qls->pls', metric, Qls_tilde)
        self.Ipq = Pls

    def compute_JK(self, C_left, C_right=None):
        """
        Compute the J and K matrices for Cocc orbitals
        """

        if C_right is None:
            C_right = C_left

        D = np.dot(C_right, C_left.T)

        J = np.zeros((self.nbf, self.nbf))
        K = np.zeros((self.nbf, self.nbf))
        core.compute_DFJK(self.Ipq, D, J, K)

        return J, K
