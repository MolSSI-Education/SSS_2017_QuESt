#!/usr/bin/env python3
import numpy as np
import quest
#from quest.core import system_energy
#from quest.core import pair_energy
import pytest

# Setup parameters
bins = 50
box_length = 10.0
reduced_density = 0.9
gr = np.zeros((bins), dtype=float)
gr_ideal = np.zeros((bins), dtype=float)
volume = np.power(box_length, 3)
delta_r = (box_length / 2.0) / bins
half_box_length = box_length / 2.0
cutoff2 = np.power(half_box_length, 2)  ##for rdf calculation, otherwise cutoff = 3*sigma
r_domain = np.linspace(0.0, half_box_length, bins)
const = 4.0 * reduced_density * np.pi / 3.0

for i_bin in range(0, bins):
    r_lower = i_bin * delta_r
    r_upper = r_lower + delta_r
    n_ideal = const * (np.power(r_upper, 3) - np.power(r_lower, 3))
    gr_ideal[i_bin] = n_ideal

number_of_snapshots = 1

coordinates = np.loadtxt(quest.lj_sample_config, skiprows=2, usecols=(1, 2, 3))
num_particles = np.shape(coordinates)[0]

@pytest.mark.skip(reason="Needs fixing")
def test_sys_ene():
    random_coordinates = (0.5 - np.random.rand(800,3)) * box_length
    energy = quest.core.system_energy(random_coordinates[:, 0], random_coordinates[:, 1], random_coordinates[:, 2], 10.0, 9.0)
    assert np.round(energy) == np.round(-4351.540194543863)


@pytest.mark.skip(reason="Needs fixing")
def test_pair_ene():
    tot = 0.0
    random_coordinates = (0.5 - np.random.rand(800,3)) * box_length
    for i_particle in range(0, num_particles):
        molenergy = quest.core.pair_energy(i_particle, random_coordinates[:, 0], random_coordinates[:, 1], random_coordinates[:, 2],
                                              10.0, 9.0)
        tot += molenergy
    assert np.round(tot / 2) == np.round(-4351.540194543863)


@pytest.mark.skip(reason="Needs fixing")
def test_rdf():
    (r, rdf, gr_max, r_max) = quest.core.rdf(delta_r, gr, coordinates, num_particles, box_length, cutoff2, gr_ideal,
                                             r_domain, number_of_snapshots)
    assert np.round(r_max) == np.round(1.020)
