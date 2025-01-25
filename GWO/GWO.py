# python implementation of Grey wolf optimization (GWO)
# minimizing rastrigin and sphere function


import random
from benchmarks import *
import copy  # array-copying convenience


# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for _ in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])

        self.fitness = fitness(self.position)  # curr fitness


# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, bounds):
    rnd = random.Random()
    minx = np.array([b[0] for b in bounds])
    maxx = np.array([b[1] for b in bounds])
    bestZ = []
    # create n random wolves
    population = [wolf(fitness, dim, minx, maxx, i) for i in range(n)]

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
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.20f" % alpha_wolf.fitness)
            print(alpha_wolf.position)

        a = 2 * (1 - Iter / max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
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
        bestZ.append(alpha_wolf.fitness)
        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position, alpha_wolf.fitness, bestZ


def main():
    num_particles = 50
    max_iter = 500
    dim = 30
    fitZ = []
    for _ in range(1):
        bounds = get_lennard_jones_bounds(30)
        best_position, best_fit, _ = gwo(lennard_jones_fitness, max_iter, num_particles, dim, bounds)
        print(best_fit)
        fitZ.append(best_fit)


if __name__ == '__main__':
    main()
