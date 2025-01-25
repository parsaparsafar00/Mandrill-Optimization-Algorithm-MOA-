import numpy as np
import random
from benchmarks import *


class Point:
    def __init__(self, position, objective_function, gender=0, dominator=None):
        self.position = position
        self.fitness = fitness_function(self.position, objective_function)
        self.is_dominator = False
        self.radius = 1.0
        self.dominator = dominator
        self.r_dev_rate = 0
        self.gender = gender
        if gender == 2:
            self.vental = random.randint(0, 1)


class Particle:
    def __init__(self, objective_function, Dim, minX, maxX, gender=0, dominator=None):
        self.is_dominator = False
        self.r_dev_rate = 0
        self.gender = gender
        self.dominator = dominator
        self.radius = 1.0
        self.rnd = random.Random()
        self.position = np.zeros(Dim)
        if self.gender == 2:
            self.vental = random.randint(0, 1)
        for i in range(Dim):
            self.position[i] = ((maxX[i] - minX[i]) * self.rnd.random()) + minX[i]
        self.fitness = fitness_function(self.position, objective_function)


def fitness_function(pos, objective_function):
    fit = objective_function(pos) + (100 * exterior_penalty_function(pos, minDimZ, maxDimZ))
    return 1 / (fit + 1e-20)


def exterior_penalty_function(x, minx, maxX):
    penalty = 0
    for i in range(len(x)):
        if x[i] < minx[i]:
            penalty += (minx[i] - x[i]) ** 2
        elif x[i] > maxX[i]:
            penalty += (x[i] - maxX[i]) ** 2
    return penalty


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_fitness_set(Set):
    Set = np.array(Set)  # Convert to NumPy array
    mean = np.mean(Set)
    std_dev = np.std(Set)

    # Add a small constant to std_dev to prevent division by very small numbers
    epsilon = 1e-20  # Choose a small constant suitable for your problem

    if std_dev < epsilon:
        std_dev = epsilon

    # Perform z-score normalization
    z_scores = (Set - mean) / std_dev
    scaled_fitness = sigmoid(z_scores)

    # Normalize the scaled fitness values
    scaled_fitness_normalized = scaled_fitness / np.sum(scaled_fitness)
    return scaled_fitness_normalized


def roulette_wheel_selection(population, global_best, alpha):
    upper_bound = 60
    lower_bound = 20
    dominators = []
    r1 = random.random()  # Random number between 0 and 1
    for j, horde in enumerate(population):
        competitors = []
        for p in horde:
            if p.gender == 0:
                p.is_dominator = False
                competitors.append(p)

        percentile = lower_bound + ((upper_bound - lower_bound) * (1 - alpha))
        # Calculate the index corresponding to the desired percentile
        index = int(np.floor(((100 - percentile) / 100) * len(competitors)))
        # Remove particles exceeding the threshold just for selecting dominator (not removing from the whole population)
        if index > 0:
            del competitors[index:]

        cumulative_probability = 0
        fit_vals = [c.fitness for c in competitors]
        fitness_values_scaled = generate_fitness_set(fit_vals)

        # Iterate through particles and select one of them based on cumulative probability
        for i in range(len(fitness_values_scaled)):
            cumulative_probability += fitness_values_scaled[i]
            if r1 <= cumulative_probability:
                c_i = horde.index(competitors[i])
                horde[c_i].is_dominator = True
                dominators.append(competitors[i])
                break  # Exit loop after selecting a dominator

        # Allocate the dominator of each of the particles inside their object
        for ind, particle in enumerate(horde):
            if particle.is_dominator:
                if global_best == particle:  # In this situation the dominator of global best should be chosen randomly
                    particle.dominator = horde[ind - 1] if horde[ind - 1] else horde[ind + 1]
                else:
                    particle.dominator = global_best
            elif particle.gender != 2:  # if the particle was not an infant
                particle.dominator = dominators[
                    -1]  # the last inserted elem to the dominants list is the dominant of the horde

    return population


def roulette_wheel_selection2(population, global_best, alpha):
    for j, horde in enumerate(population):
        for p in horde:
            p.is_dominator = False

        # Allocate the dominator of each of the particles inside their object
        for ind, particle in enumerate(horde):
            if global_best == particle:  # In this situation the dominator of global best should be chosen randomly
                particle.dominator = horde[ind - 1] if horde[ind - 1] else horde[ind + 1]
            else:
                particle.dominator = global_best

    return population


def calculate_relative_deserve(horde):
    final_fits = []
    for p1 in horde:
        particle_fit = []
        for p2 in horde:
            if p1 != p2:  # Exclude the dominator itself
                particle_fit.append(p1.fitness)
        if len(particle_fit) == 0:  # Handling the scenario of having a single element list.
            final_fits = [p1.fitness]
        else:
            final_fits.append(sum(particle_fit))  # Sum of fitness to other particles
    return final_fits


def particle_choose_criteria(current_fit, new_fit, alpha):
    # Handling negative fitness values
    if current_fit < 0 or new_fit < 0:
        arr = generate_fitness_set([current_fit, new_fit])
        current_fit = arr[0]
        new_fit = arr[1]

    x = new_fit / current_fit
    if x < 1:
        lower_bound = 2
        upper_bound = 128
        steep = lower_bound + ((upper_bound - lower_bound) * (1 - alpha))
        prob = x ** steep
    else:
        prob = 1

    rand = random.random()  # Random number between 0 and 1
    return rand < prob


def radius_generator(pop, attr, vental_attract, alpha):
    for k, horde in enumerate(pop):
        for i, particle in enumerate(horde):
            dominator = particle.dominator
            distance_to_dominator = np.linalg.norm(np.array(particle.position) - np.array(dominator.position))
            # print("radius : " + str(distance_to_dominator) + " gender:" + str(particle.gender) + " is leader:" + str(particle.is_dominator))
            attract = 1
            if particle.gender != 2:
                if particle.gender != dominator.gender:
                    attract = attr
            elif particle.vental:  # if infant was a vental
                attract = vental_attract  # vental attract is considered 10 times bigger than usual to mother
            r = alpha * ((distance_to_dominator * (1 / attract)) + (distance_to_dominator * particle.r_dev_rate))
            particle.radius = r
    return pop


def find_revolve_degrees(center_coords, point_coords):
    # Calculate the vector from the center of the sphere to the arbitrary point
    vec_to_point = np.array(point_coords) - np.array(center_coords)
    norm_vec_to_point = vec_to_point / np.linalg.norm(vec_to_point)

    theta_radians = np.arccos(norm_vec_to_point[2])  # angle with respect to the z-axis
    phi_radians = np.arctan2(norm_vec_to_point[1], norm_vec_to_point[0])  # angle with respect to the x-axis

    return theta_radians, phi_radians


def generate_new_position(particle, target_point, beta):
    center_coords = particle.position
    if target_point == particle:  # global best particle
        theta_rotation, phi_rotation = 0, 0
    else:
        theta_rotation, phi_rotation = find_revolve_degrees(center_coords,
                                                            target_point.position)  # Finding the Revolving Degree
    max_phi = beta * 180
    radii = np.random.uniform(0, particle.radius)
    phi = np.random.uniform(0, max_phi)
    azimuthal_angle = np.random.uniform(0, 360)
    newX = np.cos(np.radians(azimuthal_angle)) * np.sin(np.radians(phi))
    newY = np.sin(np.radians(azimuthal_angle)) * np.sin(np.radians(phi))
    newZ = np.cos(np.radians(phi))
    cartesian_coord = radii * np.array([newX, newY, newZ])
    rotated_point = rotate_point(theta_rotation, phi_rotation, cartesian_coord, particle.radius)
    new_position = center_coords + rotated_point
    return new_position


def generate_new_position2(particle, target_point, beta):
    return (np.array(particle.position) + np.array(target_point.position)) / 2


def rotate_point(theta_rad, phi_rad, point, r):
    # Extract coordinates of the point
    x, y, z = point
    current_theta = np.arccos(z / r)
    current_phi = np.arctan2(y, x)

    # Compute the new polar and azimuthal angles
    new_theta = current_theta + theta_rad
    new_phi = current_phi + phi_rad

    # Convert the new spherical coordinates back to Cartesian coordinates
    new_x = r * np.sin(new_theta) * np.cos(new_phi)
    new_y = r * np.sin(new_theta) * np.sin(new_phi)
    new_z = r * np.cos(new_theta)

    return new_x, new_y, new_z


def deviation_rate_allocation(population):
    # Assigning the deviation rate values based on the fitness values.
    particles = [particle for horde in population for particle in horde]
    fit_set = [p.fitness for p in particles]
    fitness_values_scaled = generate_fitness_set(fit_set)
    mean_fit = np.mean(fitness_values_scaled)
    for horde_index, horde in enumerate(population):
        for particle_index, particle in enumerate(horde):
            fit = fitness_values_scaled[horde_index * len(horde) + particle_index]
            particle.r_dev_rate = mean_fit - fit
    return population


def direction_persuasion(t, T):
    return -((t / T) ** 0.5) + 1


def particles_initialization(n, horde_num, objective_function, minX, maxX):
    pop = []
    for _ in range(horde_num):
        num_infants = int(n * 0.3)
        num_males = int(n * 0.2)
        num_females = int(n * 0.5)
        remainder = n - (num_infants + num_males + num_females)

        if remainder > 0:
            num_infants += remainder

        segments = [np.linspace(minX[i], maxX[i], n + 1).tolist() for i in range(dim)]
        population = []
        mothers = []

        for i in range(n):
            if i < num_males:
                gender = 0
                dom = None
            elif i < num_females + num_males:
                gender = 1
                dom = None
            else:
                gender = 2
                dom = mothers[i - (num_females + num_males)]

            minZ = [segments[0][i]]
            maxZ = [segments[0][i + 1]]
            for j in range(dim - 1):
                minZ.append(minX[j + 1])
                maxZ.append(maxX[j + 1])

            population.append(Particle(objective_function, dim, minZ, maxZ, gender, dom))

            # Add new female to the mothers list
            if gender == 1:
                mothers.append(population[-1])
        population = sorted(population, key=lambda temp: temp.fitness)
        pop.append(population)
    return pop


def hypersphere_optimization(objective_function, minX, maxX, max_iter, num_hordes, p_in_each_horde=20,
                             radius_deviation_enable=True,
                             oga=1.0, vental_attract=10):
    # Initialization Phase
    new_population = particles_initialization(p_in_each_horde, num_hordes, objective_function, minX, maxX)
    global_best = new_population[0][0]
    Iter = 0

    while Iter < max_iter:

        alpha = 1 - (Iter / max_iter)
        beta = direction_persuasion(Iter, max_iter)
        new_population = roulette_wheel_selection(new_population, global_best, alpha)

        if radius_deviation_enable:
            new_population = deviation_rate_allocation(new_population)
        new_population = radius_generator(new_population, oga, vental_attract, alpha)

        for i, horde in enumerate(new_population):
            for j, particle in enumerate(horde):

                dominator = particle.dominator
                new_particle = Point(generate_new_position2(particle, dominator, beta), objective_function,
                                     particle.gender, dominator)
                print("prev fit : " + str(1 / (particle.fitness + 1e-20)), " with pos : " + str(particle.position) +
                      " new fit : " + str(1 / (new_particle.fitness + 1e-20)) + " with pos : " +
                      str(new_particle.position) + " with dominator : " + str(dominator.position),
                      " and radius : ", particle.radius)
                if particle_choose_criteria(horde[j].fitness, new_particle.fitness, alpha):
                    horde[j] = new_particle
                    if particle == global_best:  # This condition is for when a particle with weaker fitness value is
                        # chosen with respect to the probs instead of the current global best particle, so we need to
                        # change the global best variable, otherwise, we will have non-existing global best variable
                        global_best = new_particle

            new_population[i] = sorted(horde, key=lambda x: x.fitness)

            # Finding the global best particle
            for s in new_population:
                for p in s:
                    gl_fit = global_best.fitness
                    p_fit = p.fitness
                    # Handling negative and positive values both...
                    if (gl_fit > 0 > p_fit) or (gl_fit < 0 < p_fit):
                        if p_fit < gl_fit:
                            global_best = p
                    elif p_fit > gl_fit:
                        global_best = p

        print(" ----------------------------- ")
        print("Global Best Position: " + str(global_best.position))
        print("Iteration: " + str(Iter) + " ==> Min Cost = %.20f" % (1 / global_best.fitness - 1e-20))
        Iter += 1

    print()
    print("The obtained solution is: " + str(global_best.position))
    return global_best.position


minDimZ = [-5, -5, -5]
maxDimZ = [5, 5, 5]
dim = len(maxDimZ)
best_position = hypersphere_optimization(quadratic_fitness, minDimZ, maxDimZ, 150, 3, 10, True,
                                         1.5)
