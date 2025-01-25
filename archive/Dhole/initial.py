import random
import math  # cos() for Rastrigin
import copy  # array-copying convenience
import sys  # max float
import matplotlib.pyplot as plt

# -------fitness functions---------

# rastrigin function
def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

def fitness_quadratic(position):
    return position[0] ** 2

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


        plt.figure(figsize=(8, 6))
        for i in range(n):
            x_values = [pos[0] for pos in wolf_positions_history[i]]
            y_values = [pos[1] for pos in wolf_positions_history[i]]
            plt.plot(x_values, y_values, label=f'Wolf {i + 1}')

        # Plot the current alpha, beta, and gamma wolves
        alpha_x, alpha_y = alpha_wolf.position[0], alpha_wolf.position[1]
        beta_x, beta_y = beta_wolf.position[0], beta_wolf.position[1]
        gamma_x, gamma_y = gamma_wolf.position[0], gamma_wolf.position[1]
        plt.scatter([alpha_x, beta_x, gamma_x], [alpha_y, beta_y, gamma_y], color='red', marker='x',
                    label='Alpha, Beta, Gamma')

        plt.title(f'Iteration {Iter} - GWO Algorithm')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()


        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter += 1
    # end-while

    # returning the best solution
    return alpha_wolf.position


# ----------------------------


# Driver code for rastrigin function

print("\nBegin grey wolf optimization on rastrigin function\n")
dim = 3
fitness = fitness_rastrigin

print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim - 1):
    print("0, ", end="")
print("0)")

num_particles = 3
max_iter = 100

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting GWO algorithm\n")

best_position = gwo(fitness, max_iter, num_particles, dim, -20.0, 20.0)

print("\nGWO completed\n")
print("\nBest solution found:")
print(["%.6f" % best_position[k] for k in range(dim)])
err = fitness(best_position)
print("fitness of best solution = %.6f" % err)

print("\nEnd GWO for rastrigin\n")
