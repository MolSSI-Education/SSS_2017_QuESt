#!/usr/bin/env python3
import numpy as np
import quest
#from quest.core import system_energy
#from quest.core import pair_energy
import pytest

# General parameters
epsilon = 1.0
sigma = 1.0
box_length = 10.0
reduced_density = 0.9
volume = np.power(box_length, 3)
cutoff2= np.power(3*sigma,2) ## uses in LJ potential calculation
num_particles = np.shape(coordinates)[0]

# rdf parameters
bins = 50
number_of_snapshots = 1


def test_sys_ene():
    random_coordinates = (0.5 - np.random.rand(800,3)) * box_length
    energy = quest.core.system_energy(random_coordinates[:, 0], random_coordinates[:, 1], random_coordinates[:, 2],
                                      10.0, cutoff2, epsilon)
    #random coordinates should be a very high number since it is not minimized. 
    assert np.round(energy) > np.round(1000)


def test_pair_ene():
    tot = 0.0
    random_coordinates = (0.5 - np.random.rand(800,3)) * box_length
    energy = quest.core.system_energy(random_coordinates[:, 0], random_coordinates[:, 1], random_coordinates[:, 2],
                                  10.0, cutoff2, epsilon)

    for i_particle in range(0, 800):
        molenergy = quest.core.pair_energy(i_particle, random_coordinates[:, 0], random_coordinates[:, 1], random_coordinates[:, 2],
                                              10.0, cutoff2, epsilon)
        tot += molenergy
    #pair energy should be twice as much energy since double counting. 
    assert np.round(tot / 2) == np.round(energy)


#need better testing
def test_rdf():
    (r, rdf, gr_max, r_max) = quest.rdf_func(coordinates, bins, box_length, reduced_density, number_of_snapshots)
    assert np.round(r_max) == np.round(1.020)
