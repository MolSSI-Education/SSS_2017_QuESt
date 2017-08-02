import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani

"""
Plot the RDF plot
"""
def plot_rdf(r_domain, gr, r_max, gr_max):
    
    fig = plt.figure()
    plt.xlabel("Distance")
    plt.ylabel("Radial Distribution Function")
    plt.style.use('seaborn-poster')
    ax = fig.add_subplot(111)
    line, = ax.plot(r_domain, gr, color='#ee8d18', lw=3)
    ax.plot([r_max], [gr_max], 'o')                                     # <--
    ax.text(r_max + .05, gr_max, 'Local max', fontsize=20)
    plt.show()
