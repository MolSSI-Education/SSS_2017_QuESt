import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani

#def plot_rdf(writer, ax, line, r_d, gr, r_max, gr_max):
def plot_rdf(writer, ax, line, r_d, gr):

    gr_max = gr.max()
    r_max = r_d[ gr.argmax() ]
    #line.set_data(r_max, gr_max)
    ax.clear()
    ax.plot(r_d, gr)
    ax.text(r_max + .05, gr_max, 'Local max', fontsize=20)
    writer.grab_frame()
    #plt.show()

#def myplot(wr, pl1, pl2, x, y, x_max, y_max):
    
#    pl1.set_data(x, y)
    
