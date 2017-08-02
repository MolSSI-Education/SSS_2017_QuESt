import numpy as np
from scf_utils import *
import scipy.sparse.linalg as spla
import scipy as sp
import os

def response(g, F, C, L, R, nocc):
    '''
    Calculates the CPHF response.
    Expects the eri 4-tensor, g,  the Fock matrix, F, the MO coeefcient matrix, C, the left and right response tensors, L and R, and the occupation number, nocc.
    '''
    nbas = F.shape[0]
    nvirt = nbas - nocc 
    opeartor_vs = 1
    if operator_vs == 0:
        E_inv_R = np.linalg.solve(E,R.T)
    elif operator_vs == 1:
        E_kappa = E_kappa_MO(F, g, nocc, nvirt)
        E_inv_R = spla.cg(E_kappa, R)
    elif operator_vs == 2:
        E_kappa = E_kappa_MO(F, g, C, get_JK, nocc, nvirt)
        E_inv_R = spla.cg(E_kappa, R)
    return np.dot(L, E_inv_R) 


