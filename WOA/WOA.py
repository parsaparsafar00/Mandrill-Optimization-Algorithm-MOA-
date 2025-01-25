import numpy as np
from benchmarks import *


class WhaleOptimization():

    def __init__(self, opt_func, constraints, nsols, b, a, a_step, maximize=False):
        self._opt_func = opt_func
        self._constraints = constraints
        self._sols = self._init_solutions(nsols)
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solutions = []

    def get_solutions(self):
        """return solutions"""
        return self._sols

    def optimize(self):
        """solutions randomly encircle, search or attack"""
        ranked_sol = self._rank_solutions()
        best_sol = ranked_sol[0]
        # include best solution in next generation solutions
        new_sols = [best_sol]

        for s in ranked_sol[1:]:
            if np.random.uniform(0.0, 1.0) > 0.5:
                A = self._compute_A()
                norm_A = np.linalg.norm(A)
                if norm_A < 1.0:
                    new_s = self._encircle(s, best_sol, A)
                else:
                    # select random sol
                    random_sol = self._sols[np.random.randint(self._sols.shape[0])]
                    new_s = self._search(s, random_sol, A)
            else:
                new_s = self._attack(s, best_sol)
            new_sols.append(self._constrain_solution(new_s))

        self._sols = np.stack(new_sols)
        self._a -= self._a_step

    def _init_solutions(self, nsols):
        """initialize solutions uniform randomly in space"""
        sols = []
        for c in self._constraints:
            sols.append(np.random.uniform(c[0], c[1], size=nsols))

        sols = np.stack(sols, axis=-1)
        return sols

    def _constrain_solution(self, sol):
        """ensure solutions are valid wrt to constraints"""
        constrain_s = []
        for c, s in zip(self._constraints, sol):
            if c[0] > s:
                s = c[0]
            elif c[1] < s:
                s = c[1]
            constrain_s.append(s)
        return constrain_s

    def _rank_solutions(self):
        """find best solution"""
        fitness = []
        for sol in self._sols:
            fitness.append(self._opt_func(np.array(sol)))
        sol_fitness = [(f, s) for f, s in zip(fitness, self._sols)]

        # best solution is at the front of the list
        ranked_sol = list(sorted(sol_fitness, key=lambda x: x[0], reverse=self._maximize))
        self._best_solutions.append(ranked_sol[0])

        return [s[1] for s in ranked_sol]



    def return_best_solutions(self):

        return (sorted(self._best_solutions, key=lambda x: x[0], reverse=self._maximize)[0])

    def _compute_A(self):
        r = np.random.uniform(0.0, 1.0, size=len(self._constraints))
        return (2.0 * np.multiply(self._a, r)) - self._a

    def _compute_C(self):
        return 2.0 * np.random.uniform(0.0, 1.0, size=len(self._constraints))

    def _encircle(self, sol, best_sol, A):
        D = self._encircle_D(sol, best_sol)
        return best_sol - np.multiply(A, D)

    def _encircle_D(self, sol, best_sol):
        C = self._compute_C()
        D = np.linalg.norm(np.multiply(C, best_sol) - sol, axis=-1)
        return D

    def _search(self, sol, rand_sol, A):
        D = self._search_D(sol, rand_sol)
        return rand_sol - np.multiply(A, D)

    def _search_D(self, sol, rand_sol):
        C = self._compute_C()
        return np.linalg.norm(np.multiply(C, rand_sol) - sol, axis=-1)

    def _attack(self, sol, best_sol):
        D = best_sol - sol  # Now D is a vector
        L = np.random.uniform(-1.0, 1.0, size=best_sol.shape)  # Ensure L matches the dimension of `best_sol`
        return np.multiply(np.multiply(np.linalg.norm(D), np.exp(self._b * L)), np.cos(2.0 * np.pi * L)) + best_sol


def main():
    bounds = get_lennard_jones_bounds(30)
    fit_func = lennard_jones_fitness
    max_iter = 500
    b = 1
    a = 2
    a_step = a / max_iter

    opt_alg = WhaleOptimization(fit_func, bounds, 50, b, a, a_step)

    for _ in range(max_iter):
        opt_alg.optimize()
        global_best_woa = opt_alg.return_best_solutions()
        print(global_best_woa)



if __name__ == '__main__':
    main()
