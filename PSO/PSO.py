# python implementation of particle swarm optimization (PSO)
# minimizing rastrigin and sphere function

import random
import math  # cos() for Rastrigin
import copy  # array-copying convenience
import sys  # max float
from benchmarks import *


# -------------------------

# particle class
class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random()

        # initialize position of the particle with 0.0 value
        self.position = [0.0 for i in range(dim)]

        # initialize velocity of the particle with 0.0 value
        self.velocity = [0.0 for i in range(dim)]

        # initialize best particle position of the particle with 0.0 value
        self.best_part_pos = [0.0 for i in range(dim)]

        # loop dim times to calculate random position and velocity
        # range of position and velocity is [minx, max]
        for i in range(dim):
            self.position[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])
        self.velocity[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])

        # compute fitness of particle
        self.fitness = fitness(self.position)  # curr fitness

        # initialize best position and fitness of this particle
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness  # best fitness


# particle swarm optimization function
def pso(fitness, max_iter, n, dim, bounds):
    minx = np.array([b[0] for b in bounds])
    maxx = np.array([b[1] for b in bounds])
    # hyper parameters
    w = 0.729  # inertia
    c1 = 1.49445  # cognitive (particle)
    c2 = 1.49445  # social (swarm)

    rnd = random.Random(0)

    # create n random particles
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    # compute the value of best_position and best_fitness in swarm
    best_swarm_pos = [0.0 for _ in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max  # swarm best

    # computer best particle of swarm and it's fitness
    for i in range(n):  # check each particle
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    # main loop of pso
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.50f" % best_swarm_fitnessVal)

        for i in range(n):  # process each particle
            # compute new velocity of curr particle
            for k in range(dim):
                r1 = rnd.random()  # randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = (
                        (w * swarm[i].velocity[k]) +
                        (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) +
                        (c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k]))
                )

                # if velocity[k] is not in [minx, max]
                # then clip it
                if swarm[i].velocity[k] < minx[k]:
                    swarm[i].velocity[k] = minx[k]
                elif swarm[i].velocity[k] > maxx[k]:
                    swarm[i].velocity[k] = maxx[k]

            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]

            # compute fitness of new position
            swarm[i].fitness = fitness(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

        # for-each particle
        Iter += 1
    return best_swarm_pos


fitness = lennard_jones_fitness
num_particles = 50
max_iter = 500

fitZ = []

bounds = get_lennard_jones_bounds(30)
best_position = pso(fitness, max_iter, num_particles, 30, bounds)
best_fit = fitness(best_position)
print(best_position)
fitZ.append(best_fit)
print(np.std(fitZ))

