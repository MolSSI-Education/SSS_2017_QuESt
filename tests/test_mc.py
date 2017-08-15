#!/usr/bin/env python3
"""
A short test for the MC module
"""
import numpy as np
import quest
import pytest

# General parameters
epsilon = 1.0
sigma = 1.0
box_length = 10.0
reduced_density = 0.9
volume = np.power(box_length, 3)
cutoff2 = np.power(3 * sigma, 2)
coordinates = np.loadtxt(quest.lj_sample_config, skiprows=2, usecols=(1, 2, 3))

# RDF parameters
bins = 50
number_of_snapshots = 1


def test_sys_ene():
    energy = quest.core.system_energy(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 10.0,
                                      cutoff2, epsilon)
    assert np.round(energy, 5) == np.round(-4351.540194543859, 5)


def test_pair_ene():
    tot = 0.0
    energy = quest.core.system_energy(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 10.0,
                                      cutoff2, epsilon)

    for i_particle in range(0, coordinates.shape[0]):
        molenergy = quest.core.pair_energy(i_particle, coordinates[:, 0], coordinates[:, 1],
                                           coordinates[:, 2], 10.0, cutoff2, epsilon)
        tot += molenergy

    # pair energy should be twice as much energy since double counting.
    assert np.round(tot / 2, 5) == np.round(energy, 5)


def test_rdf():
    (r, rdf, gr_max, r_max) = quest.rdf_func(coordinates, bins, box_length, reduced_density,
                                             number_of_snapshots)
    assert np.round(r_max, 3) == 1.020
