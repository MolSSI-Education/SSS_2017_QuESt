#!/usr/bin/env python3
import numpy as np
import mc_demo as mc
bins=50
box_length = 10.0
reduced_density = 0.9
gr = np.array(bins * [0], dtype = float)
gr_ideal = np.array(bins * [0], dtype = float)
volume = np.power(box_length, 3)
delta_r = (box_length / 2.0) / bins
half_box_length = box_length / 2.0  # half of the box_length is considered for calculation
half_box_length2 = np.power(half_box_length, 2)
r_domain = np.linspace(0.0, half_box_length, bins)
const = 4.0 * reduced_density * np.pi / 3.0
for i_bin in range(0, bins):
    r_lower = i_bin * delta_r
    r_upper = r_lower + delta_r
    n_ideal = const * (np.power(r_upper,3) - np.power(r_lower,3))
    gr_ideal[i_bin] = n_ideal

number_of_snapshots =1



coordinates = np.loadtxt("lj_sample_config_periodic1.txt", skiprows=2, usecols=(1, 2, 3))


energy = mc.system_energy(coordinates[:,0], coordinates[:,1], coordinates[:,2], 10.0, 9.0)
print(energy)
tot =0.0
for i_particle in range(0, 800):
	molenergy = mc.pair_energy(i_particle, coordinates[:,0],coordinates[:,1], coordinates[:,2], 10.0 , 9.0)
	tot += molenergy
print (tot / 2.0)
gr = mc.rdf(bins,  delta_r, coordinates[:,0],coordinates[:,1], coordinates[:,2], 10.0 , 9.0)
print (gr)
