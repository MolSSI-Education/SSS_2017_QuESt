import numpy as np
from . import mc_demo as mc

def tail_correction(box_length, cutoff, num_particles):
   """
   This is correction function to the potential energy. The correction function
   is used to compensate the switch off the LJ potential after rcutoff
   """
   """
   The function is U*corr = (8*pi*num_particles^2)/(3*volume)*[(1/3)*(1/cutoff)^9-(1/cutoff)^3]
   """
   #defined the box length for a cubic box, cutoff = 3*sigma,
   #where sigma is the distance where the LJ potential is zero

   try:
       isinstance(box_length, float) and isinstance(cutoff, float)
   except:
       raise ValueError('Box length and cutoff must be float')

   if box_length < 0.0 or cutoff < 0.0:
       raise ValueError('Box length and cutoff must be positive')

   volume = box_length**3
   sig_by_cutoff3 = (1 / cutoff)**3
   sig_by_cutoff9 = sig_by_cutoff3**3
   e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
   e_correction *= 8.0 / 9.0 * np.pi * num_particles**2 / volume
   return e_correction

def rdf_func(delta_r, gr, coordinates_NIST, num_particles, box_length, cutoff2, gr_ideal, r_domain, number_of_snapshots):
    """
        THis function calculates the radial distribution function for a system of LJ particles
        gr_ideal normalizes the distribution obtained by couting the number of particles within a shell of thickness deltr_r
        number_of_snapshots to take the average
        The rcutoff2 is square of the half of the box length upto which the rdf is calculated
        r_domain is the x axis of the rdf plot 
    """
    gr = mc.rdf(delta_r, gr, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], box_length , cutoff2)
    current_gr = gr / gr_ideal / number_of_snapshots / num_particles
    gr_max = np.amax(current_gr)
    r_domain_index = np.argmax(current_gr)
    r_domain_max = r_domain[r_domain_index]
    return(r_domain, current_gr, gr_max, r_domain_max)

