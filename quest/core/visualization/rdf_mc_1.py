
# coding: utf-8

# In[1]:

# Modules

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import plotRDF as pla

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
plt.xlim(0, 2.5)
plt.ylim(0, 2)
writer.setup(fig, "writer_test.mp4", 200)

plt.xlabel("Distance")
plt.ylabel("Radial Distribution Function")
plt.style.use('seaborn-poster')
ax = fig.add_subplot(111)



# In[2]:

# Parameters

reduced_density = 0.9
reduced_temperature = 0.9
num_particles = 100

beta = 1 / reduced_temperature
box_length = np.cbrt(num_particles / reduced_density)
# box_length = 10.0
# cutoff = 3.0
cutoff = box_length / 2.0
max_displacement = 0.1
cutoff2 = np.power(cutoff, 2)


# In[3]:

# Generate initial state

# Randomly placing particles in a box
# coordinate = (0.5 - np.random.rand(num_particles, 3)) * box_length

# Adding particles in a lattice. There might be many more
# ways to do this!
spacing = int(np.cbrt(num_particles) + 1)
x_vector = np.linspace(0.0, box_length, spacing)
y_vector = np.linspace(0.0, box_length, spacing)
z_vector = np.linspace(0.0, box_length, spacing)
grid  = np.meshgrid(x_vector, y_vector, z_vector)
stack = np.vstack(grid)
coordinates = stack.reshape(3, -1).T
excess = len(coordinates) - num_particles
coordinates = coordinates[:-excess]
coordinates *= 0.95

# Reading a reference configuration from NIST
# coordinates = np.loadtxt("lj_sample_config_periodic1.txt", skiprows=2, usecols=(1,2,3))


# In[4]:

# Lennard Jones potential implementation

def lennard_jones_potential(rij2):
    sig_by_r6 = np.power(1 / rij2, 3)
    sig_by_r12 = np.power(sig_by_r6, 2)
    return 4.0 * (sig_by_r12 - sig_by_r6)


# In[5]:

# Minimum image distance implementation

def minimum_image_distance(r_i, r_j, box_length):
    rij = r_i - r_j
    rij = rij - box_length * np.round(rij / box_length)
    rij2 = np.dot(rij, rij)
    return rij2


# In[6]:

# Computation of the total system energy

def total_potential_energy(coordinates, box_length):
    e_total = 0.0

    for i_particle in range(0, num_particles):
        for j_particle in range(0, i_particle):
            r_i = coordinates[i_particle]
            r_j = coordinates[j_particle]
            rij2 = minimum_image_distance(r_i, r_j, box_length)
            if rij2 < cutoff2:
                e_pair = lennard_jones_potential(rij2)
                e_total += e_pair
    return e_total



# In[7]:

# Computation of the energy tail correction

def tail_correction(box_length):
    volume = np.power(box_length, 3)
    sig_by_cutoff3 = np.power(1.0 / cutoff, 3)
    sig_by_cutoff9 = np.power(sig_by_cutoff3, 3)
    e_correction = sig_by_cutoff9 - 3.0 * sig_by_cutoff3
    e_correction *= 8.0 / 9.0 * np.pi * num_particles / volume * num_particles
    return e_correction


# In[8]:

def get_molecule_energy(coordinates, i_particle):
    e_total = 0.0
    i_position = coordinates[i_particle]
    for j_particle in range(0, num_particles):
        if i_particle != j_particle:
            j_position = coordinates[j_particle]
            rij2 = minimum_image_distance(i_position, j_position, box_length)

            if rij2 < cutoff2:
                e_pair = lennard_jones_potential(rij2)
                e_total += e_pair
    return e_total


# In[9]:

total_pair_energy = total_potential_energy(coordinates, box_length)
tail_correction = tail_correction(box_length)


# In[14]:

# # Monte Carlo algorithm
n_trials = 0
n_accept = 0
n_steps = 2000
energy_array = np.zeros(n_steps)

## RDF stuff
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

line, = ax.plot(r_domain, gr, color='#ee8d18', lw=3)

for i_bin in range(0, bins):
    r_lower = i_bin * delta_r
    r_upper = r_lower + delta_r
    n_ideal = const * (np.power(r_upper,3) - np.power(r_lower,3))
    gr_ideal[i_bin] = n_ideal

for i_step in range(0,n_steps):
    n_trials += 1
    i_particle = np.random.randint(num_particles)
    random_displacement = (2.0 * np.random.rand(3) - 1.0)* max_displacement
    old_position = coordinates[i_particle].copy()
    old_energy = get_molecule_energy(coordinates, i_particle)
    coordinates[i_particle] += random_displacement
    new_energy = get_molecule_energy(coordinates, i_particle)

    delta_e = new_energy - old_energy

    if delta_e <= 0.0:
        accept = True
    else:
        random_number = np.random.rand(1)
        p_acc = np.exp(-beta*delta_e)
        if random_number < p_acc:
            accept = True
        else:
            accept = False

    if accept == True:
        n_accept += 1
        total_pair_energy += delta_e
    else:
        coordinates[i_particle] = old_position

    if np.mod(i_step + 1,100) == 0:
        acc_rate = float(n_accept) / float(n_trials)
        if (acc_rate < 0.380):
            max_displacement *= 0.8
        elif (acc_rate > 0.42):
            max_displacement *= 1.2
        n_trials = 0
        n_accept = 0


        if (i_step > 200):
            for i_mol in range(0, num_particles):
                for j_mol in range(i_mol + 1, num_particles):
                    r_i = coordinates[i_mol]
                    r_j = coordinates[j_mol]
                    rij2 = minimum_image_distance(r_i, r_j, box_length)
                    if (rij2 < half_box_length2):
                        rij = np.sqrt(rij2)
                        bin_number = int(rij / delta_r)
                        gr[bin_number] += 2

            current_gr = gr / gr_ideal / number_of_snapshots / num_particles
            number_of_snapshots += 1
            pla.plot_rdf(writer, ax, line, r_domain, current_gr) 

            #pl2.set_data(r_domain, current_gr)
            #writer.grab_frame()


writer.finish()
# In[ ]:

# plt.plot(energy_array[50000:])
