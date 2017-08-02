import numpy as np
import matplotlib.pyplot as plt

def lj_plot(fnam, mol):
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


