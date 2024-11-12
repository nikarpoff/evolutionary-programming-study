import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from time import time
from random import randint, random


def count_mutation_delta(generation, total_generations, limit, b):
    # Random number [0, 1)
    r = random()

    degree = (1 - generation / total_generations) ** b
    return limit ** (1 - r ** degree)


class RealGeneticAlgorithm:
    """
    Solves task of finding maximum of target function (fitness function) using genetic algorithm.
    :param fitness_function: target function
    :param n: number of dimensions
    :param population_size: size of the population
    :param max_generations: number of max available generations. For a situation when it is impossible to find a satisfying optimum
    :param crossover_probability: probability of a crossover
    :param mutation_probability: probability of a mutation
    :param epsilon: stop algorithm when  mean fitness function is less than or equal to this value
    """

    def __init__(self, fitness_function, n: int, population_size: int, max_generations: int,
                 crossover_probability: float,
                 mutation_probability: float, left_edge: float, right_edge: float, epsilon=0.05):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.epsilon = epsilon
        self.n = n

        # Every chromosome is vector with n elements (xi). Initially uniformly randomized.
        self.population = np.random.uniform(left_edge, right_edge, size=(population_size, n))
        self.fitness_values = np.zeros(population_size, float)

    def start(self):
        generation = 1  # generations counter
        last_fitness_mean = 0.  # mean fitness
        mean_fitness_delta = 1.  # delta between current and previous mean fitness function value

        start_time = time()

        while generation < self.max_generations:
            # 1. Evaluate fitness.
            self.evaluate()

            # 2. Reproduction.
            self.reproduction()

            # 3. Crossing.
            self.crossover()

            # 4. Mutate.
            self.mutate(generation, 1., 1., 0.8)

            mean_fitness_delta = self.fitness_values.mean() - last_fitness_mean
            last_fitness_mean = self.fitness_values.mean()

            print(f'Generation [{generation}/{self.max_generations}]\t\t' +
                  f'Fitness delta = {mean_fitness_delta:.3f}\t\tMin fitness = {min(self.fitness_values):.3f}\t\t' +
                  f'Mean: {(sum(self.fitness_values) / self.population_size):.3f}\n')

            if self.n == 2:
                self.evaluate()
                self.draw_plot(generation)

            generation += 1

        end_time = time()

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Found answer: {min(self.fitness_values):4f}. ',
              f'Required generations: {generation}. Fitness mean delta: {abs(mean_fitness_delta):3f}\n',
              f'n = {self.n}, Pc = {self.crossover_probability}, Pm = {self.mutation_probability}')

    def evaluate(self):
        """
        Evaluates the fitness of the population.
        """
        for i in range(self.population_size):
            # Count fitness of every chromosome in population
            self.fitness_values[i] = self.fitness_function(self.population[i])

    def reproduction(self):
        """
        Roulette based reproduction algorithm.
        """
        fitness_copy = self.fitness_values.copy()
        fitness_copy_inverted = 1 / fitness_copy

        total_fitness = sum(fitness_copy_inverted)
        probabilities = [fitness / total_fitness for fitness in fitness_copy_inverted]
        new_population = []

        for i in range(self.population_size):
            current_wheel_probability = 0.
            random_probability = random()

            # Emulate roulette pass.
            for j in range(self.population_size):
                current_wheel_probability += probabilities[j]

                if random_probability < current_wheel_probability:
                    # Random chosen probability in range [last_chromosome_p, new_chromosome_p).
                    new_population.append(self.population[j])
                    break

            # If there is case with rounding error (last chromosome wasn't added) add last chromosome manually.
            if i != len(new_population) - 1:
                new_population.append(self.population[-1])

        # Update population.
        for i in range(self.population_size):
            self.population[i] = new_population[i]

    def crossover(self):
        """
        Randomly distributes population on pairs and makes crossing (with probability).
        """
        random_indexes = np.random.permutation(self.population_size)

        for i in range(0, len(random_indexes) - 1, 2):
            # With probability.
            if random() < self.crossover_probability:
                # Generate uniform random number (0, 1)
                u = 0.
                while u == 0.:
                    u = random()

                degree = 1 / (self.n + 1)

                if u > 0.5:
                    beta = (1 / (2 * (1 - u))) ** degree
                else:
                    beta = (2 * u) ** degree

                p1 = self.population[random_indexes[i]]
                p2 = self.population[random_indexes[i + 1]]

                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

                self.population[random_indexes[i]] = c1
                self.population[random_indexes[i + 1]] = c2

    def mutate(self, generation, left_limit, right_limit, b):
        """
        Non-uniform Mutation for population with probability
        :param generation: number of generation
        :param left_limit: left boundary of mutate value
        :param right_limit: right boundary of mutate value
        :param b: degree of an influence from generations number
        """
        for i in range(self.population_size):
            if random() < self.mutation_probability:
                # Dimension (component of chromosome) to mutate.
                rand_dimension = randint(0, self.n - 1)
                chromosome_component = self.population[i][rand_dimension]

                # Random direction to mutate.
                direction = randint(0, 1)

                if direction == 1:
                    delta = count_mutation_delta(generation, self.max_generations, right_limit, b)
                    self.population[i][rand_dimension] = chromosome_component + delta
                else:
                    delta = count_mutation_delta(generation, self.max_generations, left_limit, b)
                    self.population[i][rand_dimension] = chromosome_component - delta

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

        x_dots = self.population[:, 0]
        y_dots = self.population[:, 1]
        z_dots = self.fitness_values

        ax.scatter3D(x_dots, y_dots, z_dots, color='green', marker='o', s=50, edgecolor='black')

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"Generation {generation}. Population size is {self.population_size}.\n" +
                     f"Pc -> {self.crossover_probability}. Pm -> {self.mutation_probability}. " +
                     f"Min fitness = {min(self.fitness_values):.4f}")

        plt.show()
