"""
A basic wavefunction class
"""

import psi4


class Wavefunction(object):
    """
    Basic wavefunction class

    Parameters
    -------
    mol : QuESt Molecule class
        Needs to have a basis set at least.
    parameters : dictionary
        Holds the starting QM parameters

    Attributes
    -------
    options : dictionary 
        Initializes with qm options from parameters file, can be edited. 
    mints : psi4.core.MintsHelper
        Psi4 mints object 
    energies : dictionary 
        Stores various energies that have been calculated (including scf,
        nuclear, mp2, etc.).
    arrays : dictionary 
        Stores various arrays (including a coefficient matrix, a fock matrix,
        etc.)

    Examples
    -------
    >>>h2o_wf = Wavefunction(mol)
    Creates an instance of the Wavefunction class called h2o_wf

    >>>self.options['ex_param'] = 5.0
    Adds a new entry to the options dictionary named "ex_param" with value 5.0
    """

    def __init__(self, mol, parameters):
        """
        Initialize the Wavefunction class.

        """

        # Set whichever options you would like. The only default is the
        # basis set name.
        self.options = parameters

        # Build the mints object
        self.mints = psi4.core.MintsHelper(mol.bas)

        # Energies: scf_energy, mp2_os_corr, mp2_ss_corr, E_nuc
        self.energies = {}

        # coefficients, fock_matrix, density,
        # and whatever else you wanna put in it... communicate about it
        self.arrays = {}
