from benchmarks import *
import numpy
from solution import solution
import time
import massCalculation
import gConstant
import gField
import move


def GSA(objf, bounds, dim, PopSize, iters):
    # GSA parameters
    ElitistCheck = 1
    Rpower = 1

    s = solution()

    """ Initializations """

    vel = numpy.zeros((PopSize, dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest = numpy.zeros(dim)
    gBestScore = 10000

    lb = numpy.array([b[0] for b in bounds])
    ub = numpy.array([b[1] for b in bounds])

    pos = numpy.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    convergence_curve = numpy.zeros(iters)

    print("GSA is optimizing  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, PopSize):

            l1 = np.clip(pos[i, :], lb, ub)
            pos[i, :] = l1
            fitness = objf(l1)
            fit[i] = fitness
            if (gBestScore > fitness):
                gBestScore = fitness
                gBest = l1

        """ Calculating Mass """
        M = massCalculation.massCalculation(fit, PopSize, M)

        """ Calculating Gravitational Constant """
        G = gConstant.gConstant(l, iters)

        """ Calculating Gfield """
        acc = gField.gField(PopSize, dim, pos, M, l, iters, G, ElitistCheck, Rpower)

        """ Calculating Position """
        pos, vel = move.move(PopSize, dim, pos, vel, acc)

        convergence_curve[l] = gBestScore

        if (l % 1 == 0):
            print(['At iteration ' + str(l + 1) + ' the best fitness is ' + str(gBestScore)])
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.Algorithm = "GSA"
    s.objectivefunc = objf.__name__
    print(gBest)
    return gBestScore


fitZ = []
for _ in range(1):
    bounds = [(0, 1) for _ in range(2)]
    best_fit = GSA(three_bar_truss_fitness, bounds, 2, 50, 500)
    fitZ.append(best_fit)
print(numpy.std(fitZ))
print(numpy.mean(fitZ))
