import numpy as np



def mp2(wfn):
    pass


def df_mp2(wfn):
    pass


def _mo_transform(g, C, nocc):
    """
    Transforms the MP2 relevant subset of the ERI tensor to MO basis
    In the future, wavefunction class will have attributes g, C, and nocc
    so the only argument will need to be a wfn class instance
    C = wfn.C
    g = wfn.g
    nocc = wfn.nocc
    """
    O = slice(None, nocc)
    V = slice(nocc, None)
    g_iajb = np.einsum('pQRS, pP -> PQRS',
        np.einsum('pqRS, qQ -> pQRS',
        np.einsum('pqrS, rR -> pqRS',
        np.einsum('pqrs, sS -> pqrS', g, C[:,V]), C[:,O]), C[:,V]), C[:,O]) 
    return g_iajb


def _denom():
    pass


def _compute_conv_e():
    pass


def _compute_DF_e():
    pass
