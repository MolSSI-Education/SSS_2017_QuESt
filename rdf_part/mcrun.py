import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import mc_demo as mc

#mc.system_energy(coordinates[:,0], coordinates[:,1], coordinates[:,2], 10.0, 9.0)
#mc.pair_energy(i_particle, coordinates[:,0],coordinates[:,1], coordinates[:,2], 10.0 , 9.0)


# Parameters
reduced_density = 0.9 
reduced_temperature = 0.9 
num_particles = 800 

beta = 1 / reduced_temperature
#box_length = np.cbrt(num_particles / reduced_density)
box_length = 10.0
cutoff = 3.0 
cutoff2 = np.power(cutoff, 2)
max_displacement = 0.1

# # Generate initial state

# # random insertion
# # coordinates = (np.random.rand(num_particles, 3) - 0.5) * box_length

# # lattice insertion
# spacing = int(np.cbrt(num_particles) + 1)
# x_vector = np.linspace(0.0, box_length, spacing)
# y_vector = np.linspace(0.0, box_length, spacing)
# z_vector = np.linspace(0.0, box_length, spacing)
# grid = np.meshgrid(x_vector, y_vector, z_vector)
# stack = np.vstack(grid)
# coordinates = stack.reshape(3, -1).T

# # avoid excess coordinates
# coordinates = coordinates[:num_particles]

# # avoid overlapping at the end
# coordinates *= 0.95


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='r', marker='o')

# input NIST coordinates
coordinates_NIST = np.loadtxt("lj_sample_config_periodic1.txt", skiprows=2, usecols=(1, 2, 3))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(coordinates_NIST[:, 0], coordinates_NIST[:, 1], coordinates_NIST[:, 2], c='r', marker='o')

def lennard_jones_potential(rij2):
    sig_by_r6 = (1 / rij2)**3
    sig_by_r12 = sig_by_r6**2
    return 4.0 * (sig_by_r12 - sig_by_r6)

def minimum_image_distance(r_i, r_j, box_length):
    rij = r_i - r_j
    rij -= box_length * np.round(rij / box_length)
    rij2 = np.dot(rij, rij)
    return rij2

def total_potential_energy(coordinates, box_length):

    e_total = 0.0

    for i_particle in range(0, num_particles):
        for j_particle in range(0, i_particle):
            rij2 = minimum_image_distance(coordinates[i_particle], coordinates[j_particle], box_length)
            if rij2 < cutoff2:
                e_pair = lennard_jones_potential(rij2)
                e_total += e_pair
    return e_total

def tail_correction(box_length):
    volume = box_length**3
    sig_by_cutoff3 = (1 / cutoff)**3
    sig_by_cutoff9 = sig_by_cutoff3**3
    e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
    e_correction *= 8.0 / 9.0 * np.pi * num_particles**2 / volume
    return e_correction

def get_molecule_energy(i_particle, coordinates, box_length):

    e_particle = 0.0

    for j_particle in range(num_particles):
        if i_particle != j_particle:
            rij2 = minimum_image_distance(coordinates[i_particle], coordinates[j_particle], box_length)
            if rij2 < cutoff2:
                e_pair = lennard_jones_potential(rij2)
                e_particle += e_pair
    return e_particle
def rdf_func(bins,coordinates_NIST, num_particles, box_length, gr_ideal, delta_r, r_domain, cutoff2,number_of_snapshots):
    gr = mc.rdf(bins,  delta_r, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], box_length , cutoff2)
    current_gr = gr / gr_ideal / number_of_snapshots / num_particles
    gr_max = np.amax(current_gr)
    r_domain_index = np.argmax(current_gr)
    r_domain_max = r_domain[r_domain_index]
    return(r_domain, current_gr, gr_max, r_domain_max)

total_pair_energy = mc.system_energy(coordinates_NIST[:,0], coordinates_NIST[:,1], coordinates_NIST[:,2], box_length, cutoff2)
#total_potential_energy(coordinates_NIST, box_length)
tail_correction = tail_correction(box_length)


# Monte Carlo algorithm

num_steps = 100000
energy_array = np.zeros(num_steps)

num_accept = 0
num_trials = 0
bins = 50
gr = np.array(bins * [0], dtype = float)
gr_ideal = np.array(bins * [0], dtype = float)
volume = np.power(box_length, 3)
delta_r = (box_length / 2.0) / bins
half_box_length = box_length / 2.0  # half of the box_length is considered for calculation
half_box_length2 = np.power(half_box_length, 2)
r_domain = np.linspace(0.0, half_box_length, bins)
const = 4.0 * reduced_density * np.pi / 3.0
number_of_snapshots = 1

for i_bin in range(0, bins):
    r_lower = i_bin * delta_r
    r_upper = r_lower + delta_r
    n_ideal = const * (np.power(r_upper,3) - np.power(r_lower,3))
    gr_ideal[i_bin] = n_ideal


for i_step in range(num_steps):
    num_trials += 1
    i_particle = np.random.randint(num_particles)
    old_position = coordinates_NIST[i_particle].copy()
    old_energy = mc.pair_energy(i_particle, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], box_length , cutoff2)
    #get_molecule_energy(i_particle, coordinates_NIST, box_length)
    random_displacement = (np.random.rand(3) - 0.5) * 2 * max_displacement
    coordinates_NIST[i_particle] += random_displacement
    new_energy = mc.pair_energy(i_particle, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], 10.0 , 9.0)
    #get_molecule_energy(i_particle, coordinates_NIST, box_length)
    delta_energy = new_energy - old_energy

    if delta_energy < 0.0:
        accept = True
    else:
        random_number = np.random.rand(1)
        p_acc = np.exp(-beta * delta_energy)
        if random_number < p_acc:
            accept = True
        else:
            accept = False

    if accept:
        num_accept += 1
        total_pair_energy += delta_energy
    else:
        coordinates_NIST[i_particle] -= random_displacement

    if np.mod(i_step +1, 1000) == 0:
        acc_rate = float(num_accept) / float(num_steps)
        num_accept = 0
        num_trials = 0
        if acc_rate < 0.38:
            max_displacement *= 0.8
        elif acc_rate > 0.42:
            max_displacement *= 1.2
        
        if (i_step > 30000):
            (r_now, gr_now, gr_max, r_max)=rdf_func(bins,coordinates_NIST, num_particles, box_length, gr_ideal, delta_r, r_domain, cutoff2,number_of_snapshots)
            number_of_snapshots += 1

#           mc.rdf(bins, delta_r, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], 10.0 , 9.0)
#            for i_mol in range(0, num_particles):
#                for j_mol in range(i_mol + 1, num_particles):
#                    r_i = coordinates_NIST[i_mol]
#                    r_j = coordinates_NIST[j_mol]
#                    rij2 = minimum_image_distance(r_i, r_j, box_length)
#                    if (rij2 < half_box_length2):
#                        rij = np.sqrt(rij2)
#                        bin_number = int(rij / delta_r)
#                        gr[bin_number] += 2

#           current_gr = gr / gr_ideal / number_of_snapshots / num_particles
#           gr_max = np.amax(current_gr)
#           r_domain_index = np.argmax(current_gr)
#           r_domain_max = r_domain[r_domain_index]
#           print(gr_max, r_domain_max)

            plt.plot(r_now, gr_now)
            plt.show()
  
    total_energy = (total_pair_energy + tail_correction) / num_particles
    energy_array[i_step] = total_energy

