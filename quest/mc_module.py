#Monte Carlo module

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

def monte_carlo(sigma, epsilon, coord_file, parameters):
    #coordinates_NIST = np.loadtxt("lj_sample_config_periodic1.txt", skiprows=2, use cols=(1, 2, 3)) 
    #coord_file ="/home/yohanna/Gits/SSS_2017_QuESt/quest/lj_sample_config_periodic1.txt"
    box_length = parameters['mm']['box_size']
    num_steps = parameters['mm']['num_steps']
    tolerance_acce_rate = parameters['mm']['acc_rate']
    max_displacement_scaling = parameters['mm']['max_displacement']
    coordinates_NIST = np.loadtxt(coord_file, skiprows=2, usecols=(1, 2, 3))        
    num_particles = len(coordinates_NIST)    
    num_accept = 0
    num_trials = 0

    for i_step in range(num_steps):
        num_trials += 1
        i_particle = np.random.randint(num_particles)
        old_position = coordinates_NIST[i_particle].copy()
        old_energy = mc.pair_energy(i_particle, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], box_length , cutoff2)
        #get_molecule_energy(i_particle, coordinates_NIST, box_length)
        random_displacement = (np.random.rand(3) - 0.5) * 2 * max_displacement
        coordinates_NIST[i_particle] += random_displacement
        new_energy = mc.pair_energy(i_particle, coordinates_NIST[:,0],coordinates_NIST[:,1], coordinates_NIST[:,2], 10.0 , 9.0, epsilon)
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
             if acc_rate < tolerance_acce_rate[0]:
                 max_displacement *= max_displacement_scaling[0]
             elif acc_rate > tolerance_acce_rate[1]:
                 max_displacement *= max_displacement_scaling[1]
        total_energy = (total_pair_energy + tail_correction) / num_particles
        energy_array[i_step] = total_energy
        print (total_energy*num_particles)
        return total_energy    
