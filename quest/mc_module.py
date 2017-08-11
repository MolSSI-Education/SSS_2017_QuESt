"""
This module will take in the sigma and epsilon parameters and will do a Monte Carlo simulation
"""

import numpy as np
from . import core


def tail_correction(box_length, cutoff, num_particles):
    """
    This is correction function to the potential energy. The correction function
    is used to compensate the switch off the LJ potential after rcutoff

    The function is U*corr = (8*pi*num_particles^2)/(3*volume)*[(1/3)*(1/cutoff)^9-(1/cutoff)^3]

    defined the box length for a cubic box, cutoff = 3*sigma,
    where sigma is the distance where the LJ potential is zero
    """

    if not (isinstance(box_length, float) and isinstance(cutoff, float)):
        raise ValueError('Box length and cutoff must be float')

    if (box_length < 0.0) or (cutoff < 0.0):
        raise ValueError('Box length and cutoff must be positive')

    volume = box_length**3
    sig_by_cutoff3 = (1 / cutoff)**3
    sig_by_cutoff9 = sig_by_cutoff3**3
    e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
    e_correction *= 8.0 / 9.0 * np.pi * num_particles**2 / volume
    return e_correction


def rdf_func(delta_r, gr, coordinates_NIST, num_particles, box_length, cutoff2, gr_ideal, r_domain,
             number_of_snapshots):
    """
    This function calculates the radial distribution function for a system of LJ particles
    gr_ideal normalizes the distribution obtained by couting the number of particles within a shell of thickness deltr_r
    number_of_snapshots to take the average
    The rcutoff2 is square of the half of the box length upto which the rdf is calculated
    r_domain is the x axis of the rdf plot
    """
    gr = core.rdf(delta_r, gr, coordinates_NIST[:, 0], coordinates_NIST[:, 1], coordinates_NIST[:, 2], box_length,
                cutoff2)
    current_gr = gr / gr_ideal / number_of_snapshots / num_particles
    gr_max = np.amax(current_gr)
    r_domain_index = np.argmax(current_gr)
    r_domain_max = r_domain[r_domain_index]
    return (r_domain, current_gr, gr_max, r_domain_max)


def lennard_jones_potential(rij2):
    sig_by_r6 = (1 / rij2)**3
    sig_by_r12 = sig_by_r6**2
    return 4.0 * (sig_by_r12 - sig_by_r6)


def minimum_image_distance(r_i, r_j, box_length):
    rij = r_i - r_j
    rij -= box_length * np.round(rij / box_length)
    rij2 = np.dot(rij, rij)
    return rij2


def tail_correction(box_length):
    volume = box_length**3
    sig_by_cutoff3 = (1 / cutoff)**3
    sig_by_cutoff9 = sig_by_cutoff3**3
    e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
    e_correction *= 8.0 / 9.0 * np.pi * num_particles**2 / volume
    return e_correction

def monte_carlo(epsilon,
                box_length,
                cutoff,
                num_steps,
                tolerance_acce_rate=[0.38, 0.42],
                max_displacement_scaling=[0.8, 1.2]):
    """
    Runs MC simulation using the MCMC algorithm.
    Conformations are chosen from a probability density based on the Metropolis Hastings criteria
    for acceptance

    ------------
    PARAMETERS
    epsilon:
        passed in from lj fitting in order to convert reduced units to real units.
    box_length:
        decides the size of the box
    cutoff:
        determines the distance in which pair-wise energies are to be calculated.
    num_steps:
        controls how many iterations the MC simulation will occur. start sampling
        after 30,000 steps to ensure equilibration.
    tolerance_acce_rate:
        controls the limits of accepting/rejecting conformational states.
    max_displacement_scaling:
        controls how much the atoms should be displaced, so that we
        obtain more conformations within our tolerance acceptance rate.

    returns:
        total energy and array of accepted coordinates.

#    """

    coordinates_NIST = np.loadtxt("lj_sample_config_periodic1.txt", skiprows=2, usecols=(1, 2, 3))
    num_particles = len(coordinates_NIST)
    coordinates_of_simulation = np.zeros((num_particles, 3))
    num_accept = 0
    num_trials = 0
    count = 0  # for storing accepted conformations

    for i_step in range(num_steps):
        num_trials += 1
        i_particle = np.random.randint(num_particles)
        old_position = coordinates_NIST[i_particle].copy()
        old_energy = core.pair_energy(i_particle, coordinates_NIST[:, 0], coordinates_NIST[:, 1], coordinates_NIST[:, 2],
                                    box_length, cutoff2)
        #get_molecule_energy(i_particle, coordinates_NIST, box_length)
        random_displacement = (np.random.rand(3) - 0.5) * 2 * max_displacement
        coordinates_NIST[i_particle] += random_displacement
        new_energy = core.pair_energy(i_particle, coordinates_NIST[:, 0], coordinates_NIST[:, 1], coordinates_NIST[:, 2],
                                    box_length, cutoff, epsilon)
        #get_molecule_energy(i_particle, coordinates_NIST, box_length)
        delta_energy = new_energy - old_energy

        if delta_energy < 0.0:
            accept = True
            coordinates_of_simulation[count] = coordinates_NIST[i_particle]
            count += 1
        else:
            random_number = np.random.rand(1)
            p_acc = np.exp(-beta * delta_energy)
            if random_number < p_acc:
                accept = True
                coordinates_of_simulation[count] = coordinates_NIST[i_particle]
                count += 1
            else:
                accept = False

        if accept:
            num_accept += 1
            total_pair_energy += delta_energy
        else:
            coordinates_NIST[i_particle] -= random_displacement

        if np.mod(i_step + 1, 1000) == 0:
            acc_rate = float(num_accept) / float(num_steps)
            num_accept = 0
            num_trials = 0
            if acc_rate < tolerance_acce_rate[0]:
                max_displacement *= max_displacement_scaling[0]
            elif acc_rate > tolerance_acce_rate[1]:
                max_displacement *= max_displacement_scaling[1]
        total_energy = (total_pair_energy + tail_correction) / num_particles
        energy_array[i_step] = total_energy
        print(total_energy * num_particles)
        return total_energy
