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
	

	plt.plot(energy_array, "r-", label ="energy")

	plt.xlabel("No. of steps")
	plt.ylabel("Total Energy (kJ/mol)")
	
	plt.title("Total energy vs steps")
	plt.legend(loc=1, fontsize= 'x-large')
	plt.show()
	return
	


from matplotlib import animation as ani

#def plot_rdf(writer, ax, line, r_d, gr, r_max, gr_max):
def plot_rdf(writer, ax, line, r_d, gr):
    """
    Plots the radial distribution function: probability vs distance(angs)

    ----------
    writer  : animation writing object
    ax      : matplotlib.figure.object
    line    : plotting object
    r_d     : r
    gr      : g(r)
    Returns
    -------
    The graph of the radial distribution function
    """
    gr_max = gr.max()
    r_max = r_d[ gr.argmax() ]
    #line.set_data(r_max, gr_max)
    ax.clear()
    ax.plot(r_d, gr)
    ax.text(r_max + .05, gr_max, 'Local max', fontsize=10)
    writer.grab_frame()
    #plt.show()

   
