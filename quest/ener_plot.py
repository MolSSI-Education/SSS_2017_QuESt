"""
This file takes the energy array and plots!!
To call the function use mc_plot(energy_array)
"""


import numpy as np
import matplotlib.pyplot as plt

def mc_energyplot(energy_array):
	

	plt.plot(energy_array, "r-", label ="energy")

	plt.xlabel("No. of steps")
	plt.ylabel("Total Energy (kJ/mol)")
	
	plt.title("Total energy vs steps")
	plt.legend(loc=1, fontsize= 'x-large')
	plt.show()
	return
	


