import numpy as np
import scipy.sparse.linalg as spla


def E_kappa_MO(F, g, nocc, nbas):
    """
    Make F_ia^kappa in MO basis

    Parameters
    ----------
    g: Numpy 4-tensor
    kappa : 1D array
    F : 2-tensor
    nocc: integer number of occupied orbitals
    nbas: number of basis functions

    Returns
    -------
    F : 1D array
    """
    def return_func_pointer(kappa):

        # Slices - with RHF reference, nelec = 2 * nocc
        occ = slice(0, nocc)
        vir = slice(nocc, nbas)

        nvirt = nbas - nocc

        kappa.reshape((nocc, nvirt))

        # Einsum pieces of Eqn 14 in response handout
        one = np.einsum('ab,jb->ja', F[occ, occ], kappa)
        two = np.einsum('ij,jb->ib', F[vir, vir], kappa)
        thr = np.einsum('jb,iajb->ia', kappa, g[occ, vir, occ, vir]) 
        fou = np.einsum('jb,ijab->ia', kappa, g[occ, vir, occ, vir])
        fiv = np.einsum('jb,bjai->ia', kappa, g[occ, vir, occ, vir])
        
        Fiakappa = - one + two + 4 * thr - fou - fiv

        Fiakappa.ravel()

        return Fiakappa
        shape=((nbas-nocc) * nocc, (nbas-nocc) * nocc)
        matvec = return_func_pointer
    return (spla.LinearOperator(shape, matvec))

def E_kappa_AO(F, g, C, get_JK, nocc, nbas):
    """
    This is to calculate the F matrix give kappa in AO basis

    """
    def return_func_pointer(kappa):

        Cocc = C[:nocc,:]
        kocc = kapp[:nocc,:]
        Focc  = F[:nocc,:nocc]
        V = np.einsum('ij,ik,kl->jl',Cocc,kocc,C,optimize=True) #Eq. (14) in the handout
        J,K = get_JK(V);

        JK = 4 * J - K - K.T; # denote the centor part of Eq.(16)
        FF = np.einsum('ij,jk->ik',Focc,kocc) - np.einsum('ij,jk->ik',kocc,F)

        FF +=np.einsum('ij,jk,lk->il',Cocc,JK,C)
    
        return FF
    return return_func_pointer




def get_E(F, g, nocc, nbas):
    """
    Get four-tensor E in a brute-force way, reshape it in to
    nocc * nvirt-by-nocc * nvirt
    """
    nvirt = nbas - nocc

    # original version
    E0 = np.zeros([nbas, nbas, nbas, nbas])
    E0 += np.einsum('ij,ab->iajb', np.eye(nbas), F)
    E0 -= np.einsum('ab,ij->iajb', np.eye(nbas), F)
    E0 += 4 * np.einsum('aijb->iajb', g) - np.einsum("jiab->iajb", g) - \
        np.einsum("ajbi->iajb", g)
    E0 = E0[:nocc, nocc:, :nocc, nocc:]
    E0 = E0.reshape(nocc * nvirt, nocc * nvirt)

    # concise version
    """
    E = -F.reshape(1, nbas, 1, nbas)[:, :nvirt, :, :nvirt] + \
        F.reshape(nbas, 1, nbas, 1)[:nocc, :, :nocc, :] + \
        4. * g[:nocc, :nvirt, :nocc, :nvirt] - \
        g[:nocc, :nocc, :nvirt, :nvirt].swapaxes(1, 2) -\
        g[:nvirt, :nocc, :nvirt, :nocc].swapaxes(0, 3).swapaxes(1, 2)
    """

    E0 = E0.reshape(nocc * nvirt, nocc * nvirt)

    return E0
