import matplotlib.pyplot as plt
import numpy as np

from time import time


class Particle:
    """Particle - solution of the task"""
    def __init__(self, n, left_edge, right_edge, p_id):
        self.p_id = p_id
        self.position = np.random.uniform(left_edge, right_edge, n)
        self.velocity = np.random.uniform(-1, 1, n)
        self.best_position = self.position.copy()
        self.best_value = float('inf')
        self.value = float('inf')


class AlgorithmPSO:
    """
    Solves task of finding maximum of target function (fitness function) using genetic algorithm.
    :param fitness_function: target function
    :param n: number of dimensions
    :param population_size: size of the population
    :param max_generations: number of max available generations
    :param c1: acceleration coefficient for cognitive component
    :param c2: acceleration coefficient for social component
    """
    def __init__(self, fitness_function, n: int, population_size: int, max_generations: int, left_edge: float,
                 right_edge: float, c1=2., c2=2., w=0.9):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.n = n
        self.c1 = c1
        self.c2 = c2
        self.w = w

        # Particles (array of population_size solutions).
        self.particles = [Particle(n, left_edge, right_edge, i) for i in range(population_size)]

        # The global best particle.
        self.best_particle = None

        # The global best value.
        self.best_value = float('inf')

        self.fitness_values = np.zeros(shape=(self.population_size, ))

    def start(self):
        generation = 0  # generations counter

        start_time = time()

        while generation < self.max_generations:
            for particle in self.particles:
                # 1. Evaluate fitness.
                particle.value = self.fitness_function(particle.position)

                # 2. Update the personal best.
                if particle.value < particle.best_value:
                    particle.best_value = particle.value
                    particle.best_position = particle.position.copy()

                # 3. Update the global best.
                if particle.value < self.best_value:
                    self.best_value = particle.value
                    self.best_particle = particle.position.copy()

            for particle in self.particles:
                # 4. Update velocities and particles.
                # Get random coefficients for randomisation steps
                r1 = np.random.uniform(0, 1, size=(self.n, ))
                r2 = np.random.uniform(0, 1, size=(self.n, ))

                # Count cognitive and social components.
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.best_particle - particle.position)

                # Update velocity and position.
                particle.velocity = self.w * particle.velocity + cognitive + social

                # Ограничение скорости
                max_velocity = (self.right_edge - self.left_edge) / 2
                particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)

                particle.position = particle.position + particle.velocity

                # Clip position if it is out of bounds.
                particle.position = np.clip(particle.position, self.left_edge, self.right_edge)
                self.fitness_values[particle.p_id] = self.fitness_function(particle.position)

            self.w = max(0.4, self.w * 0.99)

            print(f'Generation [{generation + 1}/{self.max_generations}]\t\t' +
                  f'Best solution = {self.best_value:.3f}\t\t Current min fitness = {min(self.fitness_values):.3f}\t\t ' +
                  f'Mean: {(sum(self.fitness_values) / self.population_size):.3f}\n')

            if self.n == 2:
                self.draw_plot(generation + 1)

            generation += 1

        end_time = time()

        # Update the best solution in last iteration.
        for particle in self.particles:
            particle.value = self.fitness_function(particle.position)

            if particle.value < particle.best_value:
                particle.best_value = particle.value
                particle.best_position = particle.position.copy()

            if particle.value < self.best_value:
                self.best_value = particle.value
                self.best_particle = particle.position.copy()

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Found answer: {self.best_value:4f}. ',
              f'Required generations: {generation}.\n',
              f'n = {self.n}, c1 = {self.c1}, c2 = {self.c2}')

    def draw_plot(self, generation):
        dots_n = 100

        x = np.linspace(self.left_edge, self.right_edge, dots_n)
        Z = np.zeros(shape=(dots_n, dots_n))

        for i in range(dots_n):
            for j in range(dots_n):
                Z[i][j] = self.fitness_function(np.array([x[i], x[j]]))

        # Plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, x)
        ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.7)

        x_dots = np.zeros(shape=(self.population_size, ))
        y_dots = np.zeros(shape=(self.population_size, ))
        z_dots = np.zeros(shape=(self.population_size, ))

        for i in range(self.population_size):
            x_dots[i] = self.particles[i].position[0]
            y_dots[i] = self.particles[i].position[1]
            z_dots[i] = self.fitness_values[i]

        ax.scatter3D(x_dots, y_dots, z_dots, color='green', marker='o', s=50, edgecolor='black')

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"Generation {generation}. Population size is {self.population_size}.\n" +
                     f"c1 -> {self.c1}. c2 -> {self.c2}. " +
                     f"Min fitness = {min(self.fitness_values):.4f}")

        plt.show()
