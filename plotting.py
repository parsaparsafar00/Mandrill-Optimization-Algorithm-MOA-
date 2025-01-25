import os
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


def plot_particles_with_function_2D(iteration, population, global_best, o_f):
    # Define the range for plotting the function
    x_min, x_max = -5, 5  # Adjust these bounds as needed
    y_min, y_max = -5, 5
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)

    # Evaluate the objective function for the contour plot
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = o_f(np.array([X[i, j], Y[i, j]]))

    # Plot the contour of the objective function
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="Objective Function Value")

    # Scatter plot of the particle positions
    posZ = []
    for horde in population:
        for particle in horde:
            posZ.append(particle.position)
    posZ = np.array(posZ)
    plt.scatter(posZ[:, 0], posZ[:, 1], color="red", label="Particles", edgecolor="black")

    # Highlight the global best position
    plt.scatter(global_best.position[0], global_best.position[1], color="yellow", s=100, label="Global Best", edgecolor="black", marker="*")

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Particle Positions at Iteration {iteration}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_particles_with_function(iteration, population, global_best, o_f, folder_name='img', file_name='particles'):
    # Create directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Determine the dimensionality of the problem
    dim = len(population[0][0].position)

    # Create the plot
    fig = go.Figure()

    # Define custom colors for each horde
    horde_colors = ['blue', 'green', 'red', 'orange', 'black']  # Define colors for each horde
    symbols = ["circle", "square", "cross", "x"]  # Define symbols for each gender

    # Add benchmark surface plot for 2D positions
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-2, 2, 250)
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

        marker_symbol = [symbols[particle.gender] for particle in horde]

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
                size=[10],  # Larger size for the global best particle
                color=horde_colors[horde_index],
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

