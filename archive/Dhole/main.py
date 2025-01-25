import random
import math
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt


def calculate_search_space_size(min_limits, max_limits):
    # Assuming min_limits and max_limits are lists representing the minimum and maximum limits for each dimension
    dimension_ranges = [max_limits[i] - min_limits[i] for i in range(len(min_limits))]
    return np.prod(dimension_ranges)


def fitness(position):
    return sum(x ** 2 for x in position)

def set_new_position(a, leaders, population, dim, i):
    exploration = False
    rnd = random.Random()
    num_leaders = len(leaders)

    # Randomly select leaders
    leader_indices = rnd.sample(range(len(population)), num_leaders)
    selected_leaders = [population[idx] for idx in leader_indices]

    A = [a * (2 * rnd.random() - 1) for _ in range(num_leaders)]
    C = [2 * rnd.random() for _ in range(num_leaders)]

    Xnew = [0.0 for _ in range(dim)]

    for j in range(dim):
        for k in range(num_leaders):
            Xnew[j] += selected_leaders[k].position[j] - A[k] * abs(
                C[k] * selected_leaders[k].position[j] - population[i].position[j])

    for j in range(dim):
        Xnew[j] /= num_leaders

    if A[0] > 1 or A[0] < -1:
        exploration = True

    return Xnew, exploration


def switch_dholes(current_particle, dim, n, a):
    rnd = random.Random()

    A = [a * (2 * rnd.random() - 1) for _ in range(n)]
    C = [2 * rnd.random() for _ in range(n)]

    Xnew = [0.0 for _ in range(dim)]

    for j in range(dim):
        for k in range(n):
            Xnew[j] += current_particle.position[j] - a * abs(alpha_wolf.position[j] - population[i].position[j])

    return Xnew


class dhole:
    def __init__(self, fitness, dim, minx, maxx):
        self.rnd = random.Random()
        self.position = [0.0 for _ in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])
        self.fitness = fitness(self.position)


# parallel computation fo the individual particles - This single particle fitness should be compare to the fitness of
# the exploitation phase in each iteration of optimization
def dhole_individual_hunting(a, minx, maxx):
    # create n random wolves
    Dhole = [dhole(fitness, dim, minx, maxx) for _ in range(n)]
    # updating each population member with the help of best three members
    Xnew, exploration = set_new_position(a, Dhole, Dhole, dim, 0)

    # If the loop did not break, the new position is not in any Tabu region
    # fitness calculation of new solution
    fnew = fitness(Xnew)
    # greedy selection
    if fnew < Dhole[0].fitness:
        Dhole[0].position = Xnew
        Dhole[0].fitness = fnew

    return Dhole[0].position, Dhole[0].fitness


# Dhole optimization (GWO)
def dhole_herd_hunting(fitness, max_iter, n, dim, minx, maxx, landa=3, c=1, simRate=0.8):
    if landa is None:
        landa = math.ceil(n / 2)

    search_space_size = calculate_search_space_size(minx, maxx)
    tabu_radius_initial = c * (n * search_space_size ** (1 / dim))

    dhole_positions_history = [[] for _ in range(n)]

    explore_positions = []
    # create n random wolves
    population = [dhole(fitness, dim, minx, maxx) for _ in range(n)]

    # On the basis of fitness values of wolves sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best landa solutions:
    leaders = copy.copy(population[: landa])

    # main loop of gwo
    Iter = 0
    switch_phase = False  # This variable is for checking whether there is a need to switch n-1 of the particles from their current position to the new exploring position
    while Iter < max_iter:

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)
        if (switch_phase):
            switch_dholes(population[0], dim, n-1, a)
        tabu_radius = tabu_radius_initial * (1 - Iter / max_iter)
        # after every 10 iterations 
        # print iteration number and best fitness value so far
        print("Iter = " + str(Iter) + " best fitness = %.15f" % leaders[0].fitness)

        # updating each population member with the help of best three members 
        for i in range(n):
            dhole_positions_history[i].append(population[i].position[:])
            Xnew, exploration = set_new_position(a, leaders, population, dim, i)

            if exploration:
                tabu_flag = True
                while tabu_flag:  # Continue until a valid position is found
                    for historical_particle in dhole_positions_history:
                        for historical_position in historical_particle:
                            distance = np.linalg.norm(np.array(Xnew) - np.array(historical_position))
                            if distance < tabu_radius:
                                # If the new position is within the Tabu region, set a new position and recheck
                                # whether it is in the tabu space
                                Xnew, exploration = set_new_position(a, leaders, population, dim, i)
                                tabu_flag = True  # Set the flag to restart the outer loop
                                break
                            else:
                                tabu_flag = False  # Reset the flag
                        if tabu_flag:
                            # If the tabu_flag is set, restart the outer loop
                            break
                    break
                fnew = fitness(Xnew)
                # greedy selection
                if fnew < population[i].fitness:
                    population[i].position = Xnew
                    population[i].fitness = fnew
                elif fnew <= population[i].fitness * simRate:
                    explore_positions.append(dhole_individual_hunting(a, minx, maxx))

            # If the outer loop completes without breaking, it means the new position is not within the Tabu region
            # of any historical position in any set.
            else:
                # If the loop did not break, the new position is not in any Tabu region
                # fitness calculation of new solution
                fnew = fitness(Xnew)
                # greedy selection
                if fnew < population[i].fitness:
                    population[i].position = Xnew
                    population[i].fitness = fnew
                    # Here we get the fitness of each of the exploration
                    # individuals and check them with the current best global value
                    for l in range(len(explore_positions)):
                        if explore_positions[l][1][i] < population[i].fitness:
                            population[i].position = explore_positions[l][1][i]
                            population[i].fitness = explore_positions[l][1][i]
                            switch_phase = True

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)
        leaders = copy.copy(population[: landa])
        Iter += 1

    return population[0].position


# Number of wolves
n = 5
# Search space bounds for each dimension
minX = [-10.0, -5.0, 0.0]  # Adjust these values based on your specific search space
maxX = [10.0, 5.0, 10.0]

# Dimension
dim = len(minX)
# Pre processing position initialization :
segment_widths = [(maxX[i] - minX[i]) / n for i in range(len(minX))]
segments = [[minX[i] + j * segment_widths[i] for i in range(len(minX))] for j in range(n + 1)]
# Create n random wolves with assigned field indices
population = [dhole(fitness, dim, segments[i - 1], segments[i]) for i in range(1, len(segments))]
