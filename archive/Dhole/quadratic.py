import random
import math  # cos() for Rastrigin
import copy  # array-copying convenience
import sys  # max float
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# -------fitness functions---------

def fitness_quadratic_3d(position):
    x, y, z = position
    return x ** 2 + y ** 2 + z ** 2


# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx):
        self.rnd = random.Random()
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position)  # curr fitness


# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx):
    wolf_positions_history = [[] for _ in range(n)]
    rnd = random.Random(0)

    # create n random wolves
    population = [wolf(fitness, dim, minx, maxx) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key=lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations
        # print iteration number and best fitness value so far

        print("Iter = " + str(Iter) + " best fitness = %.18f" % alpha_wolf.fitness)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            wolf_positions_history[i].append(population[i].position[:])
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] += X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j] /= 3.0

            # fitness calculation of new solution
            fnew = fitness(Xnew)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key=lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position


# ----------------------------

print("\nBegin grey wolf optimization on 3D quadratic function (f(x, y, z) = y^2 + x^2 + z^2)\n")
dim = 3  # Dimension is 3 for f(x, y, z) = y^2 + x^2 + z^2
fitness = fitness_quadratic_3d

print("Goal is to minimize the 3D quadratic function in 3 variables")

num_particles = 10
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting GWO algorithm\n")

best_position = gwo(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nGWO completed\n")
print("\nBest solution found:")
print("x, y, z = [" + ", ".join("%.6f" % coord for coord in best_position) + "]")
err = fitness(best_position)
print("Minimum value of f(x, y, z) = y^2 + x^2 + z^2 at x, y, z = [" + ", ".join(
    "%.6f" % coord for coord in best_position) + "]")
