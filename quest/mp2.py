"""MP2 energy functions

Notes
-----
This module provides functions for computing MP2 corrected energies. The module
provides two interfaces mp2, and df_mp2 for computing conventional and density
fitted MP2 energies respectively.

"""
import numpy as np


def mp2(wavefunction):
    """MP2 energy using conventional ERIs

    Parameters
    ----------
    wavefunction: Wavefunction object
        Input `wavefunction` should have MO coefficients and orbital energies
        in `wavefunction.arrays` dictionary.

    Returns
    --------
    mp2_energy: float
        The mp2 energy, scf_energy + mp2 correlation energy
    wavefunction: Wavefunction object
        On return this will be the input `wavefunction` with MP2 energy
        quantities added to the `wavefunction.energies` dictionary.

    """
    g_ao = np.asarray(wavefunction.mints.ao_eri())
    orbital_energies = np.asarray(wavefunction.arrays['eps'])
    C = np.asarray(wavefunction.arrays['C'])
    num_occ_orbs = int(wavefunction.options["nel"] / 2)
    g_mo = _mo_transform(g_ao, C, num_occ_orbs)
    D = _denom(orbital_energies, num_occ_orbs)

    mp2_cor_e = _compute_conv_e(g_mo, D)
    scf_e = wavefunction.energies['scf_energy']
    mp2_total_e = scf_e + mp2_cor_e
    wavefunction.energies['mp2_correlation_e'] = mp2_cor_e
    wavefunction.energies['mp2_e'] = mp2_total_e
    return mp2_total_e


def df_mp2(wavefunction):
    """MP2 energy using Density fitted ERIs

    Parameters
    ----------
    wavefunction: Wavefunction object
        Input `wavefunction` should have MO coefficients and orbital energies
        in `wavefunction.arrays` dictionary.

    Returns
    --------
    mp2_energy: float
        The mp2 energy, scf_energy + mp2 correlation energy
    wavefunction: Wavefunction object
        On return this will be the input `wavefunction` with MP2 energy
        quantities added to the `wavefunction.energies` dictionary.
    """
    nocc = wavefunction.nocc
    orbital_energies = wavefunction.arrays['eps']
    occ_orbital_energies = orbital_energies[:nocc]
    vir_orbital_energies = orbital_energies[nocc:]
    # TODO: How should I get this?
    DF_3index = None
    mp2_cor_e = _compute_DF_e(occ_orbital_energies, vir_orbital_eergies, DF_3index)
    scf_e = wavefunction.energies['scf_e']
    mp2_total_e = mp2_cor_e + scf_e
    wavefunction.energies['mp2_correlation_e'] = mp2_cor_e
    wavefunction.energies['mp2_e'] = mp2_total_e
    return mp2_total_e


def _mo_transform(g, C, nocc):
    """Transform ERIs to the MO basis

    Parameters
    ----------
    C: numpy array
        The MO coefficients
    g: numpy array
        The AO basis, 4 index ERI tensor
    nocc:
        The number of occupied orbitals

    Returns
    -------
    g_iajb: numpy array
        The MO basis, 4 index ERI tensor. Shape will be (nocc, nvir, nocc,
        nvir)
    """
    O = slice(None, nocc)
    V = slice(nocc, None)
    g_iajb = np.einsum('pQRS, pP -> PQRS',
                       np.einsum('pqRS, qQ -> pQRS',
                                 np.einsum('pqrS, rR -> pqRS', np.einsum('pqrs, sS -> pqrS', g, C[:, V]), C[:, O]),
                                 C[:, V]), C[:, O])
    return g_iajb


def _denom(eps, nocc):
    """Build the energy denominators 1/(eps_i + eps_j - eps_a - eps_b)

    Parameters
    ----------
    eps: numpy array
        The orbital energies, shape (nbf,)

    nocc: int
        The number of occupied orbitals

    Returns
    -------
    e_denom: numpy array
        The energy denominators with shape matching the MO ERIs (nocc, nvir,
        nocc, nvir)
    """
    #get energies from fock matrix)
    #multiply C and F to get
    #must pull nocc from wavefunction when implemented
    eocc = eps[:nocc]
    evir = eps[nocc:]
    e_denom = 1 / (eocc.reshape(-1, 1, 1, 1) - evir.reshape(-1, 1, 1) + eocc.reshape(-1, 1) - evir)
    return e_denom


def _compute_conv_e(g_iajb, e_denom):
    # OS = (ia|jb)(ia|jb)/(ei+ej-ea-eb)
    os = np.einsum("iajb,iajb,iajb->", g_iajb, g_iajb, e_denom)
    # build (ib|ja)
    g_ibja = g_iajb.swapaxes(1, 3)
    # SS = [(ia|jb)-(ib|ja)](ia|jb)/(ei+ej-ea-eb)
    # print(g_iajb.shape, g_iajb.shape, e_denom.shape)
    ss = np.einsum("iajb,iajb,iajb->", (g_iajb - g_ibja), g_iajb, e_denom)
    # opposite spin, same spin
    E_mp2 = os + ss
    return E_mp2


def _compute_DF_e(eps_occ, eps_vir, Qov):
    """Computes the density fitted mp2 energy

    Parameters
    ----------
    eps_occ: numpy array
        The occupied slice of the orbital energies

    eps_vir: numpy array
        The virtual slice of the orbital energies

    Qov: numpy array
        The 3-index transformed integral tensor

    Returns
    -------
    dfmp2_corr: float
        The dfmp2 correlation energy
    """
    vv_denom = -eps_vir.reshape(-1, 1) - eps_vir

    MP2corr_OS = 0.0
    MP2corr_SS = 0.0

    for i in range(nocc):
        eps_i = eps_occ[i]
        i_Qv = Qov[:, i, :].copy()
        for j in range(i, nocc):
            eps_j = eps_occ[j]
            j_Qv = Qov[:, j, :]
            tmp = np.dot(i_Qv.T, j_Qv)
            if i == j:
                div = 1.0 / (eps_i + eps_j + vv_denom)
            else:
                div = 2.0 / (eps_i + eps_j + vv_denom)

            MP2corr_OS += np.einsum('ab,ab,ab->', tmp, tmp, div)
            MP2corr_SS += np.einsum('ab,ab,ab->', tmp - tmp.T, tmp, div)

    dfmp2_corr = MP2corr_SS + MP2corr_OS
    return dfmp2_corr
