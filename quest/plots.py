"""
This file takes the energy array and plots!!
To call the function use mc_plot(energy_array)
"""


import numpy as np
import matplotlib.pyplot as plt

def mc_energyplot(energy_array):

    """
        Plots the energy array.
        Parameters
        ----------
        energy_array : numpy array 
        Returns
        -------
        Plot 
        Examples
        --------
    energy_array = np.arange(1000)
        mc_energyplot(energy_array)
    """
    

    plt.plot(energy_array, "r-", label="energy")

    plt.xlabel("No. of steps")
    plt.ylabel("Total Energy (kJ/mol)")
    
    plt.title("Total energy vs steps")
    plt.legend(loc=1, fontsize= 'x-large')
    plt.show()
    


def plot_rdf(r_domain, gr, r_max, gr_max):
    """
    Plots the radial distribution function: probability vs distance(angs)

    ----------
    r_domain: distance
    gr      : g(r)
    r_max : r value of local maximum
    gr_max : gr value of local maximum
    Returns
    -------
    The graph of the radial distribution function
    """
    
    fig = plt.figure()
    plt.xlabel("Distance")
    plt.ylabel("Radial Distribution Function")
    plt.style.use('seaborn-poster')
    ax = fig.add_subplot(111)
    line, = ax.plot(r_domain, gr, color='#ee8d18', lw=3)
    ax.plot([r_max], [gr_max], 'o')                                     # <--
    ax.text(r_max + .05, gr_max, 'Local max', fontsize=20)
    plt.show()

