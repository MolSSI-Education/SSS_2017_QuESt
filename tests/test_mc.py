#!/usr/bin/env python3
import numpy as np
import quest
import pytest

# General parameters
epcilon = 1.0
sigma = 1.0
box_length = 10.0
reduced_density = 0.9
volume = np.power(box_length, 3)
cutoff2= np.power(3*sigma,2) ## uses in LJ potential calculation
coordinates = np.loadtxt(quest.lj_sample_config, skiprows=2, usecols=(1, 2, 3))
num_particles = np.shape(coordinates)[0]

# rdf parameters
bins = 50
number_of_snapshots = 1


def test_sys_ene():
    energy = quest.core.system_energy(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], box_length, cutoff2, epcilon)
    assert np.round(energy) == np.round(-4351.540194543863)


def test_pair_ene():
    tot = 0.0
    for i_particle in range(0, num_particles):
        molenergy = quest.core.pair_energy(i_particle, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                                              box_length, cutoff2, epcilon)
        tot += molenergy
    assert np.round(tot / 2) == np.round(-4351.540194543863)


#need better testing
def test_rdf():
    (r, rdf, gr_max, r_max) = quest.rdf_func(coordinates, bins, box_length, reduced_density, number_of_snapshots)
    assert np.round(r_max) == np.round(1.020)
