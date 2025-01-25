import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from benchmarks import *


class CSO:

    def __init__(self, fitness, P=150, n=2, pa=0.25, beta=1.5, bound=None, plot=False, min=True, verbose=False, Tmax=300):

        self.fitness = fitness
        self.P = P
        self.n = n
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.plot = plot
        self.min = min
        self.verbose = verbose

        # X = (U-L)*rand + L (U AND L ARE UPPER AND LOWER BOUND OF X)
        # U AND L VARY BASED ON THE DIFFERENT DIMENSION OF X

        self.X = []

        if bound is not None:
            for (U, L) in bound:
                x = (U - L) * np.random.rand(P, ) + L
                self.X.append(x)
            self.X = np.array(self.X).T
        else:
            self.X = np.random.randn(P, n)

    def update_position_1(self):

        num = gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        den = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        σu = (num / den) ** (1 / self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.n)
        v = np.random.normal(0, σv, self.n)
        S = u / (np.abs(v) ** (1 / self.beta))

        # DEFINING GLOBAL BEST SOLUTION BASED ON FITNESS VALUE

        for i in range(self.P):
            if i == 0:
                self.best = self.X[i, :].copy()
            else:
                self.best = self.optimum(self.best, self.X[i, :])

        Xnew = self.X.copy()
        for i in range(self.P):
            Xnew[i, :] += np.random.randn(self.n) * 0.01 * S * (Xnew[i, :] - self.best)
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def update_position_2(self):

        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.P):
            d1, d2 = np.random.randint(0, 5, 2)
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i, j] += np.random.rand() * (Xold[d1, j] - Xold[d2, j])
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def optimum(self, best, particle_x):

        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best

    def clip_X(self):

        # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE

        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[:, i] = np.clip(self.X[:, i], xmin, xmax)

    def execute(self):

        self.fitness_time, self.time = [], []
        result = []
        for t in range(self.Tmax):
            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            result.append(self.fitness(self.best))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1), 7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.best), 7))
        if self.plot:
            self.Fplot()
        return result

    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH

        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()


def main():
    dim = 2
    bounds = [(0, 1) for _ in range(dim)]
    # constraints = [(-5, 5) for _ in range(dim)]
    CSO(fitness=three_bar_truss_fitness, n=dim, Tmax=500, P=50, bound=bounds).execute()


if __name__ == '__main__':
    main()
