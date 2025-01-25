import numpy as np


def quadratic_fitness(x):  # Done
    x = np.array(x)
    Sum = 0
    for i in range(len(x)):
        Sum += ((x[i]) ** 2)
    return Sum


def quartic_noise_fitness(x):
    x = np.array(x)
    return np.sum(x ** 4) + np.random.uniform(0, 1)


def step_fitness(x):
    x = np.array(x)
    return np.sum((x + 0.5) ** 2)


def dixon_and_price_fitness(x):  # Done
    n = len(x)
    s1 = 0
    for j in range(2, n + 1):
        s1 += j * ((2 * x[j - 1] ** 2 - x[j - 2]) ** 2)
    y = s1 + (x[0] - 1) ** 2
    return y


def perm_fitness(x):  # Done
    n = len(x)
    perm = 0
    for i in range(1, n + 1):
        inner_sum = 0
        for j in range(1, n + 1):
            inner_sum += (j ** i + 10) * (((x[j - 1] / j) ** i) - 1) ** 2
        perm += inner_sum
    return perm


def rosenbrock_fitness(position):  # Done
    return sum(100 * (position[i + 1] - position[i] ** 2) ** 2 + (1 - position[i]) ** 2 for i in range(len(position) - 1))


def trid_fitness(x):
    n = len(x)
    s1 = sum([(xi - 1) ** 2 for xi in x])
    s2 = sum([x[j] * x[j - 1] for j in range(1, n)])
    return s1 - s2


def weierstrass_fitness(x, a=0.5, b=3, kmax=20):
    x = np.array(x)
    W = 0
    for k in range(kmax):
        W += np.sum(a ** k * np.cos(b ** k * np.pi * x))
    return W


def levy_fitness(x):  # Done
    n = len(x)
    z = [1 + (xi - 1) / 4 for xi in x]
    s = np.sin(np.pi * z[0]) ** 2
    for i in range(n - 1):
        s += (z[i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * z[i] + 1)) ** 2)
    s += (z[n - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * z[n - 1])) ** 2)
    return s


def levy_shifted_fitness(x):  # Done
    x = np.array(x)
    n = len(x)
    x = x + 1
    z = [1 + (xi - 1) / 4 for xi in x]
    s = np.sin(np.pi * z[0]) ** 2
    for i in range(n - 1):
        s += (z[i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * z[i] + 1)) ** 2)
    s += (z[n - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * z[n - 1])) ** 2)
    return s


def griewank_fitness(x):  # *
    x = np.array(x)
    indices = np.arange(1, len(x) + 1)
    return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(indices))) + 1


def happy_cat_fitness(x, alpha=1 / 8):
    x = np.asarray(x)  # Ensure x is a numpy array
    d = len(x)  # Dimension of the input vector

    norm_x_sq = np.sum(x ** 2)  # ||x||^2
    term1 = (norm_x_sq - d) ** 2
    term2 = (1 / d) * (0.5 * norm_x_sq + np.sum(x))
    result = (term1 ** alpha) + term2 + 0.5

    return result


def rastrigin_fitness(x):  # Done
    x = np.array(x)
    dimSize = len(x)
    A = 10
    Sum = A * dimSize
    for i in range(dimSize):
        Sum += x[i] ** 2 - A * np.cos(2 * np.pi * x[i])
    return Sum


def eggholder_fitness(x):  # 2D
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(np.sqrt(abs(x[0] - x[1] - 47)))


def kowalik_fitness(x):
    b = np.array([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0,
                  1 / 6.0, 1 / 8.0, 1 / 10.0, 1 / 12.0, 1 / 14.0,
                  1 / 16.0])
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                  0.0627, 0.0456, 0.0342, 0.0323, 0.0235,
                  0.0246])

    y = 0.0
    for i in range(11):
        bb = b[i] * b[i]
        t = a[i] - (x[0] * (bb + b[i] * x[1]) / (bb + b[i] * x[2] + x[3]))
        y += t * t
    return y


def styblinski_fitness(x):  # *
    dim = len(x)
    term = 0
    for i in range(dim):
        term += (x[i] ** 4 - 16 * x[i] ** 2 + 5 * x[i])
    return 0.5 * term + 200


def zakharov_fitness(x):  # Done
    n = len(x)
    s1 = s2 = 0
    for j in range(n):
        s1 += x[j] ** 2
        s2 += 0.5 * j * x[j]
    return s1 + s2 ** 2 + s2 ** 4


def gold_stein_function(x):
    x = np.array(x)
    X, Y = x[0], x[1]
    return (1 + (X + Y + 1) ** 2 * (19 - 14 * X + 3 * X ** 2 - 14 * Y + 6 * X * Y + 3 * Y ** 2)) * (30 + (2 * X - 3 * Y) ** 2 * (18 - 32 * X + 12 * X ** 2 + 48 * Y - 36 * X * Y + 27 * Y ** 2))


def shekel_foxholes_fitness(X):  # 2D *
    X = np.array(X)
    # Matrix 'a' as specified
    a = np.array([
        [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
        [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
    ])

    if X.ndim == 1:
        sum_part = 0
        for j in range(25):
            inner_sum = np.sum((X - a[:, j]) ** 2)
            sum_part += 1 / (j + 1 + inner_sum)
    else:  # If X is a grid of points (2D array)
        sum_part = np.zeros(X.shape[1:])
        for j in range(25):
            inner_sum = np.sum((X - a[:, j].reshape(-1, 1, 1)) ** 2, axis=0)
            sum_part += 1 / (j + 1 + inner_sum)

    fX = (1 / 500 + sum_part) ** (-1)

    return fX


def powell_fitness(x):  # Done
    return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4


def hartmann_fitness_1(x):  # 3D
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 10 ** (-4) * np.array([[3689, 1170, 2673],
                               [4699, 4387, 7470],
                               [1091, 8732, 5547],
                               [381, 5743, 8828]])
    outer_sum = 0
    for i in range(4):
        inner_sum = 0
        for j in range(3):
            inner_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        outer_sum -= alpha[i] * np.exp(-inner_sum)
    return outer_sum


def hartmann_fitness_2(xx):  # 6D

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            xj = xx[j]
            Aij = A[i, j]
            Pij = P[i, j]
            inner += Aij * (xj - Pij) ** 2
        outer += alpha[i] * np.exp(-inner)

    y = -(2.58 + outer) / 1.94
    return y


def shubert_fitness(x):  # Negative fitness values
    x = np.array(x)
    inner_sum = 0
    inner_sum2 = 0
    for j in range(1, 6):
        inner_sum += j * np.cos((j + 1) * x[1] + j)
    for j in range(1, 6):
        inner_sum2 += j * np.cos((j + 1) * x[0] + j)
    result = inner_sum * inner_sum2
    return result


def generalized_penalized_1_fitness(x, a=10, k=100, m=4):
    def u(x, a, k, m):
        """
        Penalty function u(x, a, k, m)
        """
        if x > a:
            return k * (x - a) ** m
        elif -a <= x <= a:
            return 0
        else:  # x < -a
            return k * (-x - a) ** m

    D = len(x)  # Dimension of the input
    pi_D = np.pi / D  # Precomputed factor

    # Calculate penalty term
    penalty = sum(u(x_i, a, k, m) for x_i in x)

    # Define y_i as specified in the formula
    y = [1 + 0.25 * (x_i + 1) for x_i in x]

    # Compute the main part of the fitness function
    main_sum = sum((y_i - 1) ** 2 * (1 + np.sin(3 * np.pi * y_i) ** 2) for y_i in y[:-1])
    end_term = (y[-1] - 1) ** 2  # Final term for y_D
    sine_term = 10 * np.sin(3 * np.pi * y[0]) ** 2  # First term

    # Combine all terms
    f12 = penalty + pi_D * (sine_term + main_sum + end_term)

    return f12


def generalized_penalized_2_fitness(x, a=5, k=100, m=4):
    def u(x, a=5, k=100, m=4):
        if x > a:
            return k * (x - a) ** m
        elif -a <= x <= a:
            return 0
        else:  # x < -a
            return k * (-x - a) ** m

    n = len(x)  # Dimension of input

    # Penalty term
    penalty_sum = sum(u(x_i, a, k, m) for x_i in x)

    # Main function terms
    sin_term = 0.1 * (np.sin(3 * np.pi * x[0]) ** 2)  # First term

    middle_sum = sum(
        (x[i] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[i + 1]) ** 2)
        for i in range(n - 1)
    )

    last_term = (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2)

    # Combine all terms
    f = sin_term + middle_sum + last_term + penalty_sum

    return f


def schwefel_2_26_fitness(x):  # Done
    x = np.array(x)
    return np.sum(- x * np.sin(np.sqrt(np.abs(x))))


def schwefel_12_function(x):
    return np.sum([np.sum(x[:j]) ** 2 for j in range(1, len(x) + 1)])


def schwefel_fitness(x):  # Done [420,420,420]
    x = np.array(x)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def branin_function(x):  # Done 2D
    PI = np.pi
    a = 1
    b = 5.1 / (4 * pow(PI, 2))
    c = 5 / PI
    r = 6
    s = 10
    t = 1 / (8 * PI)
    x1 = x[0]
    x2 = x[1]
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def schwefel_2_22_fitness(x):
    def prod(x):
        result = 1
        for i in x:
            result *= i
        return result

    d = len(x)
    return sum(abs(x[i]) for i in range(d)) + abs(prod(x))


def styblinski_tang_fitness(x):
    x = np.array(x)
    return (np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2) + 100


def michalewicz_fitness(x):  # 2-5-10 D
    x = np.array(x)
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi)) ** 20)


def easom_fitness(x):  # Done [pi, pi] 2D
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)


def elliptic_fitness(x):  # Done - high dim
    n = len(x)
    coefficients = np.array([10 ** (6 * (i / (n - 1))) for i in range(n)])
    return np.sum(coefficients * x ** 2)


def alpine_fitness(x):  # Done - high dim
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


def kursawe_fitness(x):
    # Objective 1

    f2 = np.sum([np.abs(x[i]) ** 0.8 + 5 * np.sin(x[i] ** 3) for i in range(len(x))])
    return f2


def shekel_fitness_10(xx):  # 4D *
    m = 10
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
    ])

    outer = 0
    for ii in range(m):
        bi = b[ii]
        inner = 0
        for jj in range(4):
            inner += (xx[jj] - C[jj, ii]) ** 2
        outer += 1 / (inner + bi)
    y = -outer
    return y


def shekel_fitness_5(xx):  # 4D
    m = 5
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
    ])

    outer = 0
    for ii in range(m):
        bi = b[ii]
        inner = 0
        for jj in range(4):
            inner += (xx[jj] - C[jj, ii]) ** 2
        outer += 1 / (inner + bi)
    y = -outer
    return y


def shekel_fitness_7(xx):  # 4D *
    m = 7
    b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
    C = np.array([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
    ])

    outer = 0
    for ii in range(m):
        bi = b[ii]
        inner = 0
        for jj in range(4):
            inner += (xx[jj] - C[jj, ii]) ** 2
        outer += 1 / (inner + bi)
    y = -outer
    return y


def six_hump_camel_fitness(x):
    x1, x2 = x
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    return term1 + term2 + term3


def keane_fitness(x):  # 2D
    x1 = x[0]
    x2 = x[1]
    a = -np.sin(x1 - x2) ** 2 * np.sin(x1 + x2) ** 2
    b = np.sqrt(x1 * x1 + x2 * x2)
    c = a / b
    return c


def de_jong_5_fitness(x):  # 2D
    A = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
    return pow(0.002 + sum([1.0 / (i + 1.0 + pow(x[0] - A[0][i], 6) + pow(x[1] - A[1][i], 6)) for i in range(25)]), -1)


def schaffer_n2(x):  # Done 2D
    x1, x2 = x[0], x[1]
    numerator = np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5
    denominator = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return 0.5 + numerator / denominator


def bent_cigar_fitness(x):  # Done
    return x[0] ** 2 + 10 ** 6 * np.sum(x[1:] ** 2)


def vincent_fitness(x, m=10):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = np.sin(x[i]) * (np.sin((i + 1) * x[i] ** 2 / np.pi)) ** (2 * m)
        result -= term
    return result + 1


def Xin_She_Yang_fitness(x):  # 2D - Done
    x1 = x[0]
    x2 = x[1]
    b = -(np.sin(x1 * x1) + np.sin(x2 * x2))
    a = (abs(x1) + abs(x2)) * np.exp(b)
    return a


def himmelblau_fitness(x):  # 2D - Done
    x = np.array(x)
    result = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    return result


def booth_fitness(x):  # 2D - Done
    x1 = x[0]
    x2 = x[1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def ackley_fitness(x, a=20, b=0.2, c=2 * np.pi):  # Done
    x = np.array(x)
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1)


# Composite Functions
# Salmon's method for generating orthogonal rotation matrices
def generate_rotation_matrix(dim):
    H = np.eye(dim)
    for i in range(dim):
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v)
        H = H - 2 * np.outer(v, v)
    return H


# Composite Function Implementation
def composite_function(x, functions, lambdas, sigmas, rotation_matrices, shifts):
    total = 0
    for i, func in enumerate(functions):
        # Apply scaling, shifting, and rotation
        x_shifted = x - shifts[i]
        x_rotated = rotation_matrices[i] @ x_shifted
        x_scaled = lambdas[i] * x_rotated
        total += sigmas[i] * func(x_scaled)
    return total


# Initialize Composite Function Parameters
def generate_composite_function(dim, selected_functions, lambdas, sigmas, shifts):
    # Parameters
    rotation_matrices = [generate_rotation_matrix(dim) for _ in range(10)]

    # Composite function wrapper
    def wrapper(x):
        return composite_function(x, selected_functions, lambdas, sigmas, rotation_matrices, shifts)

    return wrapper


def F32(dim):
    funcs = [quadratic_fitness, quadratic_fitness, quadratic_fitness, quadratic_fitness,
             quadratic_fitness, rastrigin_fitness, rastrigin_fitness, rastrigin_fitness,
             rastrigin_fitness, rastrigin_fitness]
    lambdas = np.array([5 / 100] * 10)
    sigmas = np.array([1] * 10)
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F33(dim):
    funcs = [quadratic_fitness, quadratic_fitness, ackley_fitness, ackley_fitness,
             ackley_fitness, ackley_fitness, ackley_fitness, rastrigin_fitness,
             rastrigin_fitness, rastrigin_fitness]
    lambdas = np.array([5 / 100] * 10)
    sigmas = np.array([1] * 10)
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F34(dim):
    funcs = [griewank_fitness] * 10
    lambdas = np.array([1] * 10)
    sigmas = np.array([1] * 10)
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F35(dim):
    lambdas = [5 / 32, 5 / 32, 1, 1, 5 / 0.5, 5 / 0.5, 5 / 100, 5 / 100, 5 / 100, 5 / 100]
    sigmas = [1, 1, 1, 1, 2, 2, 10, 10, 10, 10]  # Example sigmas for each function
    funcs = [happy_cat_fitness, happy_cat_fitness, rastrigin_fitness, rastrigin_fitness,
             levy_shifted_fitness, levy_shifted_fitness, griewank_fitness, griewank_fitness,
             quadratic_fitness, quadratic_fitness]
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F36(dim):
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    lambdas = [1 / 5, 1 / 5, 0.5 / 0.5, 0.5 / 0.5, 5 / 100, 5 / 100, 5 / 32, 5 / 32, 5 / 100, 5 / 100]
    sigmas = [1, 1, 2, 2, 10, 10, 3, 3, 10, 10]
    funcs = [rastrigin_fitness, rastrigin_fitness, levy_shifted_fitness, levy_shifted_fitness,
             griewank_fitness, griewank_fitness, happy_cat_fitness, happy_cat_fitness,
             quadratic_fitness, quadratic_fitness]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F37(dim):
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    lambdas = [0.02, 0.04, 3, 4, 0.025, 0.03, 0.1, 0.125, 0.045, 0.05]
    sigmas = [1, 1, 2, 2, 10, 10, 5, 5, 8, 8]
    funcs = [rastrigin_fitness, rastrigin_fitness, levy_shifted_fitness, levy_shifted_fitness,
             griewank_fitness, griewank_fitness, happy_cat_fitness, happy_cat_fitness,
             quadratic_fitness, quadratic_fitness]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F38(dim):
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    lambdas = [0.02, 0.04, 3, 4, 0.025, 0.03, 0.1, 0.125, 0.045, 0.05]
    sigmas = [1, 1, 2, 2, 10, 10, 5, 5, 8, 8]
    funcs = [rastrigin_fitness, rastrigin_fitness, levy_shifted_fitness, levy_shifted_fitness,
             griewank_fitness, griewank_fitness, happy_cat_fitness, happy_cat_fitness,
             ackley_fitness, ackley_fitness]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F39(dim):
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    lambdas = [0.02, 0.04, 3, 4, 0.025, 0.03, 0.1, 0.125, 0.045, 0.05]
    sigmas = [1, 1, 2, 2, 10, 10, 5, 5, 8, 8]
    funcs = [rastrigin_fitness, rastrigin_fitness, levy_shifted_fitness, levy_shifted_fitness,
             griewank_fitness, griewank_fitness, happy_cat_fitness, happy_cat_fitness,
             weierstrass_fitness, weierstrass_fitness]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def F40(dim):
    shifts = [np.random.uniform(-5, 5, dim) for _ in range(10)]
    lambdas = [0.02, 0.04, 3, 4, 0.025, 0.03, 0.1, 0.125, 0.045, 0.05]
    sigmas = [1, 1, 2, 2, 10, 10, 5, 5, 8, 8]
    funcs = [ackley_fitness, ackley_fitness, levy_shifted_fitness, levy_shifted_fitness,
             griewank_fitness, griewank_fitness, happy_cat_fitness, happy_cat_fitness,
             weierstrass_fitness, weierstrass_fitness]
    comp_f = generate_composite_function(dim, funcs, lambdas, sigmas, shifts)
    return comp_f


def welded_beam_fitness(x):
    # Design variables
    x0, x1, x2, x3 = x

    # Constants
    P = 6000
    L = 14
    t_max = 13600
    s_max = 30000

    # Objective functions
    f1 = 1.10471 * x0 ** 2 * x1 + 0.04811 * x2 * x3 * (14.0 + x1)
    f2 = 2.1952 / (x3 * x2 ** 3)

    # Intermediate calculations for constraints
    R = np.sqrt(0.25 * (x1 ** 2 + (x0 + x2) ** 2))
    M = P * (L + x1 / 2)
    J = 2 * np.sqrt(0.5) * x0 * x1 * (x1 ** 2 / 12 + 0.25 * (x0 + x2) ** 2)
    t1 = P / (np.sqrt(2) * x0 * x1)
    t2 = M * R / J
    t = np.sqrt(t1 ** 2 + t2 ** 2 + t1 * t2 * x1 / R)
    s = 6 * P * L / (x3 * x2 ** 2)
    P_c = 64746.022 * (1 - 0.0282346 * x2) * x2 * x3 ** 3

    # Constraints
    g1 = (t - t_max) / t_max
    g2 = (s - s_max) / s_max
    g3 = (x0 - x3) / (5 - 0.125)
    g4 = (P - P_c) / P

    # Combine objectives and constraints
    objectives = [f1, f2]
    constraints = [g1, g2, g3, g4]

    # Penalize constraints that are violated
    penalty = sum(max(0, g) ** 2 for g in constraints)

    # Return a single scalar fitness value (e.g., weighted sum of objectives + penalty)
    return objectives[0] + objectives[1] + 1e6 * penalty


def pressure_vessel_fitness(x):
    # Design variables
    x0, x1, x2, x3 = x

    # Convert variables for scaling
    d1 = x0
    d2 = x1
    r = x2
    L = x3

    # Objective function: Minimize the cost of the pressure vessel
    f = (0.6224 * d1 * r * L) + (1.7781 * d2 * r ** 2) + (3.1661 * d1 ** 2 * L) + (19.84 * d1 ** 2 * r)

    # Constraints
    g1 = -d1 + 0.0193 * r
    g2 = -d2 + 0.00954 * r
    g3 = -np.pi * r ** 2 * L - (4 / 3) * np.pi * r ** 3 + 1296000  # Volume constraint
    g4 = L - 240  # Length constraint

    constraints = [g1, g2, g3, g4]
    penalty = sum(max(0, g) ** 2 for g in constraints)
    # Return a single scalar fitness value (objective + penalty)
    return f + 1e6 * penalty


def tension_compression_spring_fitness(x):
    # Decision variables
    x1, x2, x3 = x[0], x[1], x[2]

    # Objective function
    f = (x3 + 2) * x2 * x1 ** 2

    # Constraints
    g1 = 1 - (x2 ** 3 * x3) / (71785 * x1 ** 4)
    g2 = (4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4)) + 1 / (5108 * x1 ** 2) - 1
    g3 = 1 - 140.45 * x1 / (x2 ** 2 * x3)
    g4 = ((x1 + x2) / 1.5) - 1

    # Combine all constraints into an array
    constraints = np.array([g1, g2, g3, g4])

    penalty = sum(max(0, g) ** 2 for g in constraints)
    # Return a single scalar fitness value (objective + penalty)
    return f + 1e6 * penalty


def three_bar_truss_fitness(x):
    P = 2
    RU = 2
    L = 100

    # Objective function
    fit = (2 * np.sqrt(2) * x[0] + x[1]) * L

    # Constraints
    G1 = (np.sqrt(2) * x[0] + x[1]) / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * P - RU
    G2 = (x[1]) / (np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * P - RU
    G3 = 1 / (x[0] + np.sqrt(2) * x[1]) * P - RU

    # Combine all constraints into an array
    constraints = np.array([G1, G2, G3])

    # Calculate penalty
    penalty = sum(max(0, g) ** 2 for g in constraints)

    # Return a single scalar fitness value (objective + penalty)
    return fit + 1e6 * penalty
def lennard_jones_fitness(x):
    """
    Calculates the fitness value for the Lennard-Jones potential problem.

    Parameters:
    x (np.ndarray): Positions of the atoms (shape: Nx3).

    Returns:
    float: Fitness value (Lennard-Jones potential energy + penalty for constraints).
    """
    # Constants
    N = len(x)  # Number of atoms (positions should be a 3D array of size N x 3)
    r_min = 0.4  # Minimum interatomic distance for constraints
    r_max = 4.0  # Maximum interatomic distance for constraints

    # Objective function: Lennard-Jones potential energy
    V = 0  # Total potential energy
    for i in range(N - 1):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(x[i] - x[j])  # Distance between atoms i and j
            V += (r_ij ** -12 - 2 * r_ij ** -6)  # Lennard-Jones potential


    # Compute gradient-based constraints for minimum and maximum bounds
    distance_constraints = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(x[i] - x[j])  # Distance between atoms i and j

            # Gradient-based constraint calculation
            grad_constraint_min = -12 * (r_ij ** -14 - r_ij ** -8) * (r_min - r_ij)
            grad_constraint_max = -12 * (r_ij ** -14 - r_ij ** -8) * (r_ij - r_max)

            # Append constraints
            distance_constraints.append(grad_constraint_min)
            distance_constraints.append(grad_constraint_max)

    # Combine all constraints into an array
    constraints = np.array(distance_constraints)

    # Penalty for violating constraints
    penalty = sum(max(0, g) ** 2 for g in constraints)

    # Calculate fitness value: potential energy + penalty
    fitness = V + 1e6 * penalty

    return fitness


def get_lennard_jones_bounds(N=30):
    """
    Generate bounds for the Lennard-Jones problem for N atoms.
    Returns a list of tuples representing the bounds for each dimension.
    """
    bounds = []

    for i in range(N):
        if i == 0:  # First variable (x-coordinate for the second atom)
            bounds.append((0, 4))
        elif i == 1:  # Second variable (y-coordinate for the second atom)
            bounds.append((0, 4))
        elif i == 2:  # Third variable (z-coordinate for the second atom)
            bounds.append((0, np.pi))
        else:  # Other variables
            lower_bound = -4 - (1 / 4) * np.floor((i - 4) / 3)
            upper_bound = 4 + (1 / 4) * np.floor((i - 4) / 3)
            bounds.append((lower_bound, upper_bound))

    return bounds
