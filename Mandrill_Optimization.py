import random
import copy
import time
import numpy as np
from benchmarks import *


start_time = time.time()
np.seterr(all='print')


class Particle:
    def __init__(self, objective_function=None, min_particle=None, max_particle=None, minX=None, maxX=None, gender=None, dominator=None, pre_def_position=None, radius=None):

        self.max_radius_space = None
        self.obj_func = objective_function
        self.min_range = minX
        self.max_range = maxX
        self.dimension = len(minX)
        if pre_def_position is not None:
            self.position = pre_def_position
            self.radius = radius
        else:
            self.is_leader = False
            self.r_dev_rate = 0
            self.gender = gender
            self.dominator = dominator
            self.radius = [1.0 for _ in range(self.dimension)]
            self.max_radius = [1.0 for _ in range(self.dimension)]
            self.min_radius = [1.0 for _ in range(self.dimension)]
            if self.gender == 2:
                self.ventral = 0
            self.position = np.zeros(self.dimension)
            for i in range(self.dimension):
                self.position[i] = ((max_particle[i] - min_particle[i]) * random.Random().random()) + min_particle[i]
        self.fitness = self.fitness_function()
        self.find_max_radius()

    def update_position(self, position):
        self.position = position
        self.fitness = self.fitness_function()
        self.find_max_radius()

    def update_dominator(self, new_dominator):
        self.dominator = new_dominator

    def update_gender(self, new_gender):
        self.gender = new_gender

    def fitness_function(self):
        return self.obj_func(self.position)

    def find_max_radius(self):
        distances = [0 for _ in range(self.dimension)]
        for i in range(self.dimension):
            distances[i] = min(abs(self.min_range[i] - self.position[i]), abs(self.max_range[i] - self.position[i]))
        self.max_radius_space = distances

    def show_info(self):
        for attr, value in self.__dict__.items():
            if not attr.startswith('__') and attr != 'obj_func' and attr != 'min_range' and attr != 'max_range' and attr != 'dimension':
                print(f"{attr}: {value}")

    def find_max_distance_to_edges(self):
        dist_set = []
        for d in range(self.dimension):
            dist_set.append(min(abs(self.position[d] - self.min_range[d]), abs(self.position[d] - self.max_range[d])))
        return min(dist_set)

    def __del__(self):
        pass


def particles_initialization(n, horde_num, objective_function, minX, maxX):
    dim = len(minX)
    total_particles = n * horde_num
    pop = []

    # Segment the first dimension into total_particles parts
    segments = [np.linspace(minX[0], maxX[0], total_particles + 1).tolist()]

    for i in range(1, dim):
        segments.append(np.linspace(minX[i], maxX[i], total_particles + 1).tolist())

    for _ in range(horde_num):
        num_infants = int(n * 0.3)
        num_males = int(n * 0.2)
        num_females = int(n * 0.3)
        num_juveniles = int(n * 0.2)
        remainder = n - (num_infants + num_males + num_females + num_juveniles)

        if remainder > 0:
            num_infants += remainder

        horde = []
        for i in range(n):

            idx = i + (_ * n)  # Calculate global index for the segment
            minZ = [segments[0][idx]]
            maxZ = [segments[0][idx + 1]]
            for j in range(1, dim):
                minZ.append(minX[j])
                maxZ.append(maxX[j])

            horde.append(Particle(objective_function, minZ, maxZ, minX, maxX, 0, None))

        horde = sorted(horde, key=lambda temp: temp.fitness)
        pop.append(horde)
    return pop


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def generate_fitness_set(Set):
    Set = np.array(Set)  # Convert to NumPy array
    scaled_fitness_normalized = Set / np.sum(Set)
    return scaled_fitness_normalized


def deviation_rate_allocation(population):
    # Assigning the deviation rate values based on the fitness values.
    for horde_index, horde in enumerate(population):
        fitness_values_scaled = generate_fitness_set([p.fitness for p in horde])
        mean_fit_horde = np.mean(fitness_values_scaled)
        for particle_index, particle in enumerate(horde):
            fit = fitness_values_scaled[particle_index]
            particle.r_dev_rate = mean_fit_horde - fit
    return population


def dominator_assignment(pop, alpha, global_best):
    # sorted horde
    for j, horde in enumerate(pop):
        mothers = []
        n = len(horde)
        num_infants = int(n * 0.4)
        num_males = int(n * 0.2)
        num_juveniles = int(n * 0.2)
        num_females = int(n * 0.2)
        remainder = n - (num_infants + num_males + num_females + num_juveniles)
        if remainder > 0:
            num_infants += remainder
        horde_leader = horde_leader_finder(horde, num_males, alpha)

        # bringing the horde leader to the top of the horde set
        horde.pop(horde.index(horde_leader))
        horde.insert(0, horde_leader)

        for i, particle in enumerate(horde):
            if i < 1 or i >= num_females + num_juveniles + num_infants + 1:

                gender = 0
                if global_best == particle:
                    while True:
                        dom = horde[random.randint(0, len(horde) - 1)]
                        if dom != global_best:
                            break
                elif particle.is_leader:
                    dom = global_best
                else:
                    dom = horde_leader

            elif i < num_females + 1:
                gender = 1
                mothers.append(particle)
                dom = horde_leader

            elif i < num_females + num_juveniles + 1:
                gender = 3
                dom = mothers[(i - (num_females + 1)) % len(mothers)]
            elif i < num_females + num_juveniles + num_infants + 1:
                gender = 2
                dom = mothers[(i - (num_females + num_juveniles + 1)) % len(mothers)]
                horde[i].ventral = alpha < random.randint(0, 1)

            horde[i].update_dominator(dom)
            horde[i].update_gender(gender)
    return pop


def horde_leader_finder(horde, n_males, alpha):
    upper_bound = 60
    lower_bound = 20
    horde_leader = None
    r1 = random.random()  # Random number between 0 and 1
    competitors = []
    for i, p in enumerate(horde):
        p.is_leader = False
        if i < n_males:
            competitors.append(p)
    percentile = lower_bound + ((upper_bound - lower_bound) * (1 - alpha))
    # Calculate the index corresponding to the desired percentile
    index = int(np.ceil(((100 - percentile) / 100) * len(competitors)))
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
            horde[c_i].is_leader = True
            horde_leader = competitors[i]
            break  # Exit loop after selecting a dominator

    # Allocate the dominator of each of the particles inside their object
    return horde_leader


def generate_radius(particle, ventral_movement, alpha):
    dist = np.abs(particle.dominator.position - particle.position)

    for d in range(len(dist)):
        if dist[d] == 0:
            print("Equal positions for dominator and particle for particle" + str(particle))
            print(particle.show_info())
            print("---------------")
            print(particle.dominator.show_info())
            return True

    if particle.gender == 2:
        attract = [ventral_movement for _ in range(particle.dimension)] if particle.ventral else [1 / ventral_movement for _ in range(particle.dimension)]
    else:
        attract = [1 for _ in range(particle.dimension)]

    random_vect = [4 * alpha * random.random() - alpha + 1 for _ in range(particle.dimension)]
    max_radius_vec = np.zeros(particle.dimension)
    for d in range(particle.dimension):
        max_radius_vec[d] = min((dist[d]) * random_vect[d] / attract[d] * (1 + particle.r_dev_rate), particle.max_radius_space[d])

    if particle.gender == 0 and not particle.is_leader:
        min_radius_vec = max_radius_vec / 2
    else:
        min_radius_vec = max_radius_vec / 10

    # Convert spherical coordinates to Cartesian coordinates
    particle.min_radius, particle.max_radius = min_radius_vec, max_radius_vec
    return False


def find_rotation_matrix(angles):
    d = len(angles)
    # Initialize the identity matrix
    planes = [(i, i + 1) for i in range(d - 1)]  # Default planes
    R = np.eye(d)
    for theta, (i, j) in zip(angles, planes):
        # Create a Givens rotation matrix for the (i, j) plane
        G = np.eye(d)
        G[i, i] = np.cos(theta)
        G[j, j] = np.cos(theta)
        G[i, j] = -np.sin(theta)
        G[j, i] = np.sin(theta)
        # Multiply the current rotation matrix with the new Givens rotation
        R = np.dot(R, G)
    return R


def generate_point_in_rotated_ellipsoid(center, radii, rotation_matrix):
    # Sample random point uniformly within a unit sphere
    unit_sphere_point = np.random.normal(size=len(radii))
    unit_sphere_point /= np.linalg.norm(unit_sphere_point)

    # Scale by radii to transform into ellipsoid's axis-aligned coordinates
    ellipsoid_point = unit_sphere_point * radii

    # Apply rotation and translate to center
    rotated_point = rotation_matrix.dot(ellipsoid_point)
    return rotated_point + center


def calculate_angles_with_axes(v):
    # Normalize the vector to avoid scaling issues
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        raise ValueError("Zero vector provided; angles are undefined.")

    # Calculate angles with each axis
    angles = []
    for i in range(len(v)):  # Loop through each axis
        axis_vector = np.zeros_like(v)
        axis_vector[i] = 1  # Create unit vector for the axis
        cos_theta = np.dot(v, axis_vector) / norm_v  # Compute cosine of angle
        angles.append(np.arccos(cos_theta))  # Append angle in radians
    return np.array(angles)


def generate_new_position(p):
    dominator = p.dominator
    center_coords = dominator.position

    if np.array_equal(p.position, center_coords):
        p.show_info()
        print("-------")
        p.dominator.show_info()
        raise ValueError("The target point should not be the particle")

    vector = center_coords - p.position
    angles = calculate_angles_with_axes(vector)
    rotation_matrix = find_rotation_matrix(angles)

    radius_vector = np.array([np.random.uniform(min_radius, max_radius) for max_radius, min_radius in zip(p.max_radius, p.min_radius)])
    point = generate_point_in_rotated_ellipsoid(center_coords, radius_vector, rotation_matrix)
    p.radius = radius_vector
    return point


def mandrill_optimization(objective_function, bounds, max_iter, num_hordes=3, p_in_each_horde=15, ventral_movement=5.0):
    # Initialization Phase

    minX = np.array([b[0] for b in bounds])
    maxX = np.array([b[1] for b in bounds])

    new_population = particles_initialization(p_in_each_horde, num_hordes, objective_function, minX, maxX)
    global_best = new_population[0][0]
    Iter = 0
    best_solZ = []
    best_solZ_pos = []
    while Iter < max_iter:
        # print(" ----------------------------- ")
        print("Iteration: " + str(Iter))
        # plot_particles_with_function(Iter, new_population, global_best, objective_function)

        alpha = 1 - (Iter / max_iter)
        # beta = beta_function(Iter, max_iter, ex_ex_intensity)
        new_population = dominator_assignment(new_population, alpha, global_best)
        new_population = deviation_rate_allocation(new_population)

        for i, horde in enumerate(new_population):
            for j, particle in enumerate(horde):
                converge = generate_radius(particle, ventral_movement, alpha)
                if converge:
                    return global_best.position, global_best.fitness, best_solZ, best_solZ_pos
                new_pos = generate_new_position(particle)
                prev_particle = copy.copy(particle)
                prev_fit = prev_particle.fitness
                temp_p = Particle(objective_function, None, None, minX, maxX, pre_def_position=new_pos, radius=prev_particle.radius)
                new_fit = temp_p.fitness
                del temp_p

                # print("prev loss: " + str(prev_fit), " with pos: " + str(prev_particle.position) + " new loss: " + f"{new_fit:.50f}" + " with pos: ", new_pos, " and radius: ", prev_particle.radius,
                #       " and gender of: ", prev_particle.gender)
                if new_fit < prev_fit:
                    particle.update_position(new_pos)

            new_population[i] = sorted(horde, key=lambda x: x.fitness, reverse=False)  # The particles are sorted by their fitness

            # Finding the global best particle
            for H in new_population:
                if H[0].fitness < global_best.fitness:
                    global_best = H[0]

        best_solZ.append(global_best.fitness)
        best_solZ_pos.append(global_best.position)
        print("Global Best Position: " + str(global_best.position))
        print("Min Cost = %.20f" % global_best.fitness)
        Iter += 1

    # print("The obtained solution is: " + str(global_best.position) + " with cost of: " + str(global_best.fitness))
    return global_best.position, global_best.fitness, best_solZ, best_solZ_pos


def main():
    # Example: Generate bounds for 30-dimensional Lennard-Jones problem
    bounds = [(-1, 1) for _ in range(10)]
    fitZ = []
    best_solution = []

    for i in range(1):
        best_solution, best_fit, res, _ = mandrill_optimization(rastrigin_fitness, bounds, 500, 4, 15, 20)
        fitZ.append(best_fit)

    print(np.std(fitZ))
    print(np.mean(fitZ))
    print(best_solution)


if __name__ == '__main__':
    main()
