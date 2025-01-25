import random
from benchmarks import *
import copy
import time
import os
import numpy as np
import plotly.graph_objects as go

start_time = time.time()


def plot_particles_with_function(iteration, population, global_best, o_f, folder_name='img', file_name='particles'):
    # Create directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Determine the dimensionality of the problem
    dim = len(population[0][0].position)

    # Create the plot
    fig = go.Figure()

    # Define custom colors for each horde
    horde_colors = ['blue', 'green', 'red']  # Define colors for each horde
    symbols = ["x", 'square', 'circle', 'cross']  # Define symbols for each gender

    if dim == 2:
        # Add benchmark surface plot for 2D positions
        x = np.linspace(-5, 5, 500)
        y = np.linspace(-5, 5, 500)
        X, Y = np.meshgrid(x, y)
        Z = o_f([X, Y])

        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.5, showscale=False))

    for horde_index, horde in enumerate(population):
        x = [particle.position[0] for particle in horde]
        y = [particle.position[1] for particle in horde]

        if dim == 2:
            z = [o_f(particle.position) for particle in horde]
        else:
            z = [particle.position[2] for particle in horde]

        marker_symbol = [
            symbols[particle.gender] if particle.gender < len(symbols) else 'circle'
            for particle in horde
        ]

        # Use 'diamond' symbol for the global best particle
        marker_symbol = ["diamond" if particle == global_best else marker_symbol[i] for i, particle in enumerate(horde)]

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            name=f'Horde {horde_index + 1}',  # Trace name
            marker=dict(
                symbol=marker_symbol,  # Set marker symbols based on gender
                size=[12 if particle == global_best else 10 for particle in horde],  # Larger size for the global best particle
                color=horde_colors[horde_index],  # Use the defined color for the horde
            )
        ))

    if dim == 2:
        fig.update_layout(
            title=f'Particle Positions at Iteration {iteration}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis=dict(title='Fitness', range=[0, 3]),
            )
        )
    else:
        fig.update_layout(
            title=f'Particle Positions at Iteration {iteration}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

    # Save the plot to the specified folder
    file_path = f'{folder_name}/{file_name}_{iteration}.html'
    fig.write_html(file_path)

    # Append custom HTML notes
    with open(file_path, 'a') as file:
        file.write(f"""
        <div style="padding: 10px; border: 1px solid #ddd; margin-top: 10px;">
            <h3>Legend:</h3>
            <ul>
                <li><span style="color: blue;">&#9679;</span> - Horde 1</li>
                <li><span style="color: green;">&#9679;</span> - Horde 2</li>
                <li><span style="color: red;">&#9679;</span> - Horde 3</li>
                <li><span style="font-size: 20px;">&#9679;</span> - Female</li>
                <li><span style="font-size: 20px;">&#9632;</span> - Male</li>
                <li><span style="font-size: 20px;">&#9670;</span> - Global Best</li>
            </ul>
            <h3>Benchmark Function:</h3>
            <p>The benchmark function used is defined as:</p>
            <p><code>{o_f.__name__}(x, y)</code></p>
        </div>
        """)


class Particle:
    def __init__(self, objective_function=None, min_particle=None, max_particle=None, minX=None, maxX=None, gender=None, dominator=None, min_scale=1e-20, pre_def_position=None):

        self.obj_func = objective_function
        self.min_scale = min_scale
        self.min_range = minX
        self.max_range = maxX
        self.dimension = len(minX)
        self.penalty_radius = False
        if pre_def_position is not None:
            self.position = pre_def_position
        else:
            self.is_leader = False
            self.r_dev_rate = 0
            self.gender = gender
            self.dominator = dominator
            self.radius = [1.0, 1, 1]
            if self.gender == 2:
                self.ventral = 0
            self.position = np.zeros(self.dimension)
            for i in range(self.dimension):
                self.position[i] = ((max_particle[i] - min_particle[i]) * random.Random().random()) + min_particle[i]
        self.fitness = self.fitness_function()

    def update_position(self, position):
        self.position = position
        self.fitness = self.fitness_function()

    def update_ventral(self, is_ventral):
        self.ventral = is_ventral

    def fitness_function(self):
        self.penalty_radius = np.zeros(self.dimension)
        penalty_radius = 0
        for d in range(self.dimension):

            if self.position[d] < self.min_range[d]:
                penalty_radius = max(self.min_range[d] - self.position[d], penalty_radius)
            elif self.position[d] > self.max_range[d]:
                penalty_radius = max(self.position[d] - self.max_range[d], penalty_radius)
            self.penalty_radius[d] = penalty_radius
        fit = self.obj_func(self.position)
        return fit

    def set_gender(self, g):
        self.gender = g

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


def particles_initialization(n, horde_num, objective_function, minX, maxX, solution_min_scale):
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

            horde.append(Particle(objective_function, minZ, maxZ, minX, maxX, 0, None, solution_min_scale))

        horde = sorted(horde, key=lambda temp: temp.fitness)
        pop.append(horde)
    return pop


def direction_persuasion(t, T, intensity):
    x = t / T
    numerator = np.exp(intensity * (2 * x - 1)) - np.exp(-intensity * (2 * x - 1))
    denominator = np.exp(intensity * (2 * x - 1)) + np.exp(-intensity * (2 * x - 1))
    return -0.5 * (numerator / denominator) + 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_fitness_set(Set):
    Set = np.array(Set)  # Convert to NumPy array

    # mean = np.mean(Set)
    # std_dev = np.std(Set)

    # Add a small constant to std_dev to prevent division by very small numbers
    # epsilon = 1e-20  # Choose a small constant suitable for your problem
    #
    # if std_dev < epsilon:
    #     std_dev = epsilon
    #
    # # Perform z-score normalization
    # z_scores = (Set - mean) / std_dev
    # scaled_fitness = sigmoid(z_scores)

    # Normalize the scaled fitness values
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


def particle_choose_criteria(current_fit, new_fit, alpha):
    x = current_fit / new_fit
    if x < 1:
        lower_bound = 2
        upper_bound = 128
        steep = lower_bound + ((upper_bound - lower_bound) * (1 - alpha))
        prob = x ** steep
    else:
        prob = 1

    return random.random() < prob


def dominator_assignment(pop, alpha, global_best):
    # sorted horde
    for horde in pop:
        mothers = []
        n = len(horde)
        num_infants = int(n * 0.3)
        num_males = int(n * 0.2)
        num_juveniles = int(n * 0.2)
        num_females = int(n * 0.2)
        remainder = n - (num_infants + num_males + num_females)
        if remainder > 0:
            num_infants += remainder
        horde_leader = horde_leader_finder(horde, num_males, alpha)

        for i, particle in enumerate(horde):
            if i < num_males:
                gender = 0
                if global_best == particle:
                    random.randint(1, len(horde))
                    dom = horde[random.randint(1, len(horde) - 1)]
                elif horde_leader == particle:
                    dom = global_best
                else:
                    dom = horde_leader
            elif i < num_females + num_males:
                gender = 1
                mothers.append(particle)
                dom = horde_leader
            elif i < num_females + num_males + num_juveniles:
                gender = 3
                dom = mothers[(i - (num_females + num_males)) % len(mothers)]
                # Add new female to the mothers list
            else:
                gender = 2
                dom = mothers[(i - (num_females + num_males + num_juveniles)) % len(mothers)]
                horde[i].update_ventral(alpha < random.randint(0, 1))
            horde[i].dominator = dom
            horde[i].gender = gender
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
    r1 = np.random.uniform(0, 1, size=particle.dimension)
    A = alpha * (2 * r1 - 1)
    # Calculate distance vector D
    dist = A * abs(particle.dominator.position - particle.position)

    attract = 1
    if particle.gender == 2:
        attract = ventral_movement if particle.ventral else 1 / ventral_movement
    particle.radius = [d * (1 + particle.r_dev_rate / attract) for d in dist]


def generate_new_position(particle):
    # Generate the random point in d dimensions using hyperspherical coordinates
    r = particle.radius
    cartesian_coords = np.zeros(particle.dimension)
    for d in range(particle.dimension):
        cartesian_coords[d] = particle.dominator.position[d] - r[d]


    return cartesian_coords


def mandrill_optimization(objective_function, minX, maxX, max_iter, num_hordes=3, p_in_each_horde=15, radius_deviation_enable=True, ventral_movement=2.0, ex_ex_intensity=3, solution_min_scale=1e-20):
    # Initialization Phase
    new_population = particles_initialization(p_in_each_horde, num_hordes, objective_function, minX, maxX, solution_min_scale)
    global_best = new_population[0][0]
    Iter = 0

    while Iter < max_iter:
        print(" ----------------------------- ")
        print("Iteration: " + str(Iter))
        # plot_particles_with_function(Iter, new_population, global_best, objective_function, folder_name='img', file_name='particles_positions')
        alpha = 1 - (Iter / max_iter)
        beta = direction_persuasion(Iter, max_iter, ex_ex_intensity)
        new_population = dominator_assignment(new_population, alpha, global_best)
        if radius_deviation_enable:
            new_population = deviation_rate_allocation(new_population)

        for i, horde in enumerate(new_population):
            for j, particle in enumerate(horde):
                generate_radius(particle, ventral_movement, alpha)
                prev_particle = copy.copy(particle)
                prev_fit = prev_particle.fitness
                new_pos = generate_new_position(prev_particle)
                temp_p = Particle(objective_function, None, None, minX, maxX, pre_def_position=new_pos)
                new_fit = temp_p.fitness
                if np.count_nonzero(temp_p.penalty_radius) != 0:
                    prev_particle.radius = temp_p.penalty_radius
                    new_pos = generate_new_position(prev_particle)
                    new_fit = Particle(objective_function, None, None, minX, maxX, pre_def_position=new_pos).fitness
                del temp_p
                print("prev loss: " + str(1 / prev_fit), " with pos: " + str(prev_particle.position) + " new loss: " + str(1 / new_fit) + " with pos: ",
                      new_pos, " and radius: ", particle.radius, " and gender of: ", prev_particle.gender)
                if particle_choose_criteria(prev_fit, new_fit, alpha):
                    particle.update_position(new_pos)
                    if particle == global_best:  # This condition is for when a particle with weaker fitness value is chosen with respect to the probs instead of the current global best particle,
                        # so we need to change the global best variable, otherwise, we will have non-existing global best variable
                        global_best = particle

            new_population[i] = sorted(horde, key=lambda x: x.fitness, reverse=True)  # The particles are sorted by their fitness

            # Finding the global best particle
            for s in new_population:
                for p in s:
                    if p.fitness < global_best.fitness:
                        global_best = p

        # print("Global Best Position: " + str(global_best.position))
        print("Min Cost = %.20f" % global_best.fitness)
        print(" ----------------------------- ")
        Iter += 1

    print()
    print("The obtained solution is: " + str(global_best.position) + " with cost of: " + str(global_best.fitness))
    return global_best.position


minDimZ = [-100 for _ in range(5)]
maxDimZ = [100 for _ in range(5)]
best_position = mandrill_optimization(quadratic_fitness, minDimZ, maxDimZ, 200, 3, 15, True, 2, 2, 1e-50)
end_time = time.time()
print("Consumed time is: " + str(end_time - start_time) + " seconds")
