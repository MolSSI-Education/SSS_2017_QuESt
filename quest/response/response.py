import numpy as np
import scipy.sparse.linalg as spla
import scipy as sp
from .hessian_builders import *

from quest import solvers


def ao_to_mo(tensor, transform):
    if len(g.shape) == 2:
        return transform.T @ tensor @ transform
    if len(g.shape) == 4:
        tensor = np.einsum('pqrs,pt->tqrs', tensor, transform)
        tensor = np.einsum('pqrs,qt->ptrs', tensor, transform)
        tensor = np.einsum('pqrs,rt->pqts', tensor, transform)
        tensor = np.einsum('pqrs,st->pqrt', tensor, transform)
        return tensor
    raise Exception("Not a 2 or 4 tensor")

def response(wfn):
    '''
    Calculates the CPHF response.
    Expects the eri 4-tensor, g,  the Fock matrix, F, the MO coeefcient matrix, C, the left and right response tensors, L and R, and the occupation number, nocc.
    '''

    ndocc = int(wfn.options["nel"] / 2)

    g = wfn.mints.ao_eri()
    F = wfn.arrays['F']

    get_JK = None # TO BE ADDED
    C = wfn.arrays['C']
    
    L_ao = wfn.mints(ao_dipoles())
    R_ao = wfn.mints(ao_dipoles())
   
    R = np.zeros((3,nocc * nvirt))
    L = np.zeros((3,nocc * nvirt))
    for i in range(3):
        L[i,:] = ao_to_mo(L_ao[i], C)[:nocc,nocc:].ravel()
        R[i,:] = ao_to_mo(R_ao[i], C)[:nocc,nocc:].ravel()
    
    F = ao_to_mo(F, C)
    g = ao_to_mo(g, C)

    nbas = F.shape[0]
    nvirt = nbas - nocc 
    opeartor_vs = 0
    E_inv_R = np.zeros((3,nocc * nvirt))
    if operator_vs == 0:
        E = get_E(F, g, nocc, nbas)
        for i in range(3):
            E_inv_R[i] = solvers.helper_PCG_direct(E,R[i])

    elif operator_vs == 1:
        E_kappa = E_kappa_MO(F, g, nocc, nbas)
        E_inv_R = spla.cg(E_kappa, R)
    elif operator_vs == 2:
        E_kappa = E_kappa_AO(F, g, C, get_JK, nocc, nbas)
        E_inv_R = spla.cg(E_kappa, R)
        
    return L @ E_inv_R
