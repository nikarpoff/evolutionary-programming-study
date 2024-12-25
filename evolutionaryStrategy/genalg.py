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


class EvolutionaryStrategy:
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

    def __init__(self, fitness_function, n: int, population_size: int, max_generations: int, offspring_size,
                 crossover_probability: float, mutation_probability: float, left_edge: float, right_edge: float,
                 sigma=0.1, k_success=10):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.offspring_size = offspring_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.n = n

        # Every chromosome is vector with n elements (xi). Initially uniformly randomized.
        self.population = np.random.uniform(left_edge, right_edge, size=(population_size, n))
        self.strategies = np.full((population_size, n), sigma)
        self.fitness_values = np.zeros(population_size, float)

        # Rule of success 1/5
        self.k_success = k_success
        self.successful_mutations = 0
        self.total_mutations = 0

    def start(self):
        generation = 1  # generations counter
        last_fitness_mean = 0.  # mean fitness
        mean_fitness_delta = 1.  # delta between current and previous mean fitness function value

        start_time = time()

        while generation < self.max_generations:
            # 1. Evaluate fitness.
            self.evaluate()

            # 2. Generate offspring
            offspring, offspring_strategies, success_flags = self.mutate_and_recombine()

            # 3. Evaluate offspring fitness
            offspring_fitness = np.array([self.fitness_function(ind) for ind in offspring])

            # 4. Update success statistics
            self.update_success_statistics(success_flags)

            # 5. Select the next generation (μ + λ strategy)
            self.select(offspring, offspring_fitness, offspring_strategies)

            # 6. Adjust mutation strength based on the success rule
            if (generation + 1) % self.k_success == 0:
                self.adjust_sigma()

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

    def mutate_and_recombine(self):
        """Generate offspring using mutation and recombination."""
        offspring = []
        offspring_strategies = []
        success_flags = []

        for _ in range(self.offspring_size):
            # Select two parents randomly
            parents_idx = np.random.choice(self.population_size, 2, replace=False)
            parent1, parent2 = self.population[parents_idx]
            strategy1, strategy2 = self.strategies[parents_idx]

            # Recombine (arithmetic mean)
            child = 0.5 * (parent1 + parent2)
            child_strategy = 0.5 * (strategy1 + strategy2)

            # Mutate child
            mutation = np.random.normal(0, child_strategy, size=self.n)
            child += mutation

            # Ensure child stays within bounds
            child = np.clip(child, self.left_edge, self.right_edge)

            # Update mutation strategy
            child_strategy *= np.exp(np.random.normal(0, 0.2, size=self.n))

            # Evaluate success
            parent_fitness = min(self.fitness_function(parent1), self.fitness_function(parent2))
            child_fitness = self.fitness_function(child)
            success_flags.append(child_fitness < parent_fitness)  # Success if fitness improved

            offspring.append(child)
            offspring_strategies.append(child_strategy)

        return np.array(offspring), np.array(offspring_strategies), np.array(success_flags)

    def update_success_statistics(self, success_flags):
        """Update statistics for success rule."""
        self.successful_mutations += np.sum(success_flags)
        self.total_mutations += len(success_flags)

    def select(self, offspring, offspring_fitness, offspring_strategies):
        """Select the top individuals for the next generation."""
        combined_population = np.vstack((self.population, offspring))
        combined_fitness = np.hstack((self.fitness_values, offspring_fitness))
        combined_strategies = np.vstack((self.strategies, offspring_strategies))

        # Select top μ individuals
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        self.fitness_values = combined_fitness[best_indices]
        self.strategies = combined_strategies[best_indices]

    def adjust_sigma(self):
        """Adjust mutation strength based on the success rule."""
        if self.total_mutations == 0:
            return

        success_rate = self.successful_mutations / self.total_mutations
        if success_rate > 0.2:
            self.strategies *= 1.22  # Increase mutation strength
        elif success_rate < 0.2:
            self.strategies *= 0.82  # Decrease mutation strength

        # Reset statistics
        self.successful_mutations = 0
        self.total_mutations = 0

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
