import JK_builder_C
import numpy as np
import psi4


def __init__(self):
	pass

def by_conventional(I, D):
	return JK_builder_C.form_JK_conventional(I, D)
	 
def by_df(Ig, D, C):
	return JK_builder_C.form_JK_df(Ig, D, C)

def calc_Ig(bas, mol, basname):
	'''mints = psi4.core.MintsHelper(orb)
	aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basname)
	zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
	Qls_tilde = mints.ao_eri(zero_bas, aux, orb, orb)
	Qls_tilde = np.squeeze(Qls_tilde)
	metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
	metric.power(-0.5, 1.e-14)
	metric = np.squeeze(metric)
	Pls = np.einsum('pq,qls->pls', metric, Qls_tilde)'''
	# Build the complementary JKFIT basis for the aug-cc-pVDZ basis (for example)
	aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other="aug-cc-pVDZ")
	# The zero basis set
	zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
	# Build instance of MintsHelper
	mints = psi4.core.MintsHelper(bas)
	# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
	Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
	Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions
	# Build & invert Coulomb metric, dimension (1, Naux, 1, Naux)
	metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
	metric.power(-0.5, 1.e-14)
	metric = np.squeeze(metric) # remove the 1-dimensions
	#TODO convert all einsums into np.dots to make code faster
	# Compute (P|ls)
	Pls = np.einsum('pq,qls->pls', metric, Qls_tilde)	
	
	return Pls
