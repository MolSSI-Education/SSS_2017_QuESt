import numpy as np
import psi4
import matplotlib.pyplot as plt


# .mol returns a psi4 geometry
# setmol can be given a string and it sets up a molecule

# atom - Input molecule (should be one atom)
# returnAll - Returns coefficients and energies/distances in addition to sigma if True
def lj_fit(molecule, returnAll=False):
    # stuff to return (change later)
    sigma = 0.5
    A = 2
    B = 3
    # set up distances for PES
    start = 2.0
    stop = 10.0
    step = 0.25
    distances = np.arange(start, stop, step)
    energies = np.zeros(distances.size)
    # set up geometry stuff -- make new geom object to avoid overwriting the old one
    #atom_str = molecule.mol().create_psi4_string_from_molecule()
    atom_str = molecule.create_psi4_string_from_molecule().splitlines()[2]
    print(atom_str)
    new_mol = molecule
    geom_string = '{:s}\n{:s} 1   {:2.5f}'
    # do MP2 on each distance in the array
    for i, distance in enumerate(distances):
        # construct molecule w/ correct distance
        mol_geom = geom_string.format(atom_str, atom_str, distance)
        #new_mol.set_geometry(mol_geom)
        # call MP2 on molecule and get energy
        # scf_wfn = scf( some arguments )
        # mp2_wfn = mp2(scf_wfn)
        energy = distance
        # add MP2 energy to energies list
        energies[i] = energy
    # doing the fit
    if(returnAll):
        return sigma, A, B, energies, distances
    else:
        return sigma

def get_coeffs(distances, energies):
    """
    Takes a list of distances and corresponding energies 
    perform a list squares fitting and 
    returns coefficients A for the r^12 term and B for the r^6 term
    """

    powers = [-12, -6]
    r_power = np.power(np.array(distances).reshape(-1,1), powers) 
    A,B = np.linalg.lstsq(r_power, energies)[0]
    return A,B

if __name__=='__main__':
    distances = [2,3,4,5,6,7]
    energies = [2,0,-1,-1,2,4]
    A,B = get_coeffs(distances, energies)
    print(A, B)


def plot_LJ(fnam, mol):
    """
    Plotting function to accompany lj_fit()
    """

    coeffs = np.zeros(2)
    sig, coeffs[0], coeffs[1], es, ds, = lj_fit(mol, True)

    powers = [-12, -6]
    x = np.power(np.array(ds).reshape(-1, 1), powers)

    # Build list of points
    fpoints = np.linspace(2, 7, 50).reshape(-1, 1)
    fdata = np.power(fpoints, powers)

    fit_energies = np.dot(fdata, coeffs)

    plt.xlim((2, 7))  # X limits
    plt.ylim((-7, 2))  # Y limits
    plt.scatter(ds, es)  # Scatter plot of the distances/energies
    plt.plot(fpoints, fit_energies)  # Fit data
    plt.plot([0,10], [0,0], 'k-')  # Make a line at 0
    plt.savefig(fnam)

