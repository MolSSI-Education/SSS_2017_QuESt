# This file defines a molecule class that returns information about the molecule.
# It also contains member functions set_geometry(str) for setting a geometry from a
#   string or psi4 geometry, and set_basis(str) for setting up a basis set with psi4.

import psi4


class Molecule(object):
    """
    A molecule class that is stores basic parameters, including
    psi4 geometry, basis set, number of electrons, and number of doubly
    occupied orbitals.

    """

    def __init__(self, mol=None, bas=None):
        """
        Initializes molecule with geometry mol and basis set bas.
        Calculate number of electrons nel and number of doubly occupied orbitals ndocc.

        Parameters
        ----------
        mol : str or psi4.core.Molecule
              psi4 geometry
        bas : str
              basis set

        Returns
        -------
        A built molecule

        Examples
        --------
        molecule = Molecule.mol()
        basis_set = Molecule.bas()
        num_electrons = Molecule.nel()
        num_doub_occ_orbs = Molecule.ndocc()

        """

        self.mol = mol
        self.bas = bas

        # Sets molecule and basis set
        if bas is not None:
            self.set_basis(bas)

        if mol is not None:
            self.set_geometry(mol)
            
        self.bas_name = psi4.core.BasisSet.name(self.bas)

        # Calculate the number of doubly occupied orbitals and the number of electrons
        self.nel = sum(self.mol.Z(n) for n in range(self.mol.natom()))
        self.nel -= self.mol.molecular_charge()

        if not (self.nel / 2.0).is_integer():
            raise ValueError("Molecule must have an even number of electrons to perform RHF.")
        self.ndocc = int(self.nel / 2.0)

    def set_geometry(self, geom_str):
        """
        Set the geometry of the molecule (by providing a psi4 style string or psi4 geometry).

        Parameters
        ----------
        geom_str : str
                   psi4-style geometry string (or psi4 geometry)

        Returns
        -------
        A psi4 geometry

        Examples
        --------
        Molecule.set_geometry("He")

        """

        if isinstance(geom_str, str):
            self.mol = psi4.geometry(geom_str)
        elif isinstance(geom_str, psi4.core.Molecule):
            self.mol = geom_str
        else:
            raise TypeError("Input molecule of type %s is not understood" % type(geom_str))

        self.mol.update_geometry()

    def set_basis(self, bas_str):
        """
        Set the basis set.

        Parameters
        ----------
        bas_str : str
                  basis set name

        Returns
        -------
        A basis set

        Examples
        --------
        Molecule.set_basis("STO-3G")

        """
        if (self.mol is None):
            raise Exception('Error: Geometry not defined')
        else:
            self.bas = psi4.core.BasisSet.build(self.mol, target=bas_str)
