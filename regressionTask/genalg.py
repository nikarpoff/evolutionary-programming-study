import matplotlib.pyplot as plt
import numpy as np

from time import time

import random


def cocomo_count(c, code_len):
    return c[0] * code_len ** c[1]


class RegressionGeneticAlgorithm:
    """
    Solves task of finding maximum of target function (fitness function) using genetic algorithm.
    :param population_size: size of the population
    :param max_generations: number of max available generations. For a situation when it is impossible to find a satisfying optimum
    :param crossover_probability: probability of a crossover
    :param mutation_probability: probability of a mutation
    """

    def __init__(self, population_size: int, max_generations: int, crossover_probability: float, mutation_probability: float):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

        self.population = np.ones((population_size, 2))

        self.losses_values = np.ones(population_size, float)
        self.mean_errors = []
        self.min_errors = []

    def fit(self, x, y_true):
        self.x = x
        self.y_true = y_true

        generation = 1  # generations counter

        start_time = time()

        while generation <= self.max_generations:
            # 1. Evaluate fitness.
            self.evaluate()

            # 2. Reproduction.
            self.reproduction()

            # 3. Crossing.
            self.crossover()

            # 4. Mutate.
            self.mutate()

            print(f'Generation [{generation}/{self.max_generations}]\t\t' +
                  f'\t\t Min ED = {min(self.losses_values):.3f}\t\t')

            generation += 1

        end_time = time()

        best_solution_idx = np.where(self.losses_values == min(self.losses_values))[0][0]
        self.best_solution = self.population[best_solution_idx]

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Result min ED: {min(self.losses_values):4f}. ',
              f'Required generations: {generation - 1}. Pc = {self.crossover_probability}, ',
              f'Pm = {self.mutation_probability}\n',
              f'Found solution: Ef={self.best_solution[0]:.2f} * L ^ {self.best_solution[1]:.2f}')

        self.draw_errors()

    def evaluate_best_solution(self, x_test, y_test):
        print("Start evaluate best solution.")

        y = cocomo_count(self.best_solution, x_test)
        ed = np.sqrt(np.mean(np.power(y_test - y, 2)))

        print("\tED: {:.2f}".format(ed))

    def evaluate(self):
        """
        Evaluates the fitness of the population.
        """
        # Count fitness of every chromosome in population.
        for i in range(self.population_size):
            y = cocomo_count(self.population[i], self.x)
            self.losses_values[i] = np.sqrt(np.mean(np.power(self.y_true - y, 2)))

        self.mean_errors.append(np.mean(self.losses_values))
        self.min_errors.append(np.min(self.losses_values))

    def reproduction(self):
        """
        Roulette based reproduction algorithm.
        """
        fitness_copy = self.losses_values.copy()
        fitness_copy_inverted = 1 / fitness_copy

        total_fitness = sum(fitness_copy_inverted)
        probabilities = [fitness / total_fitness for fitness in fitness_copy_inverted]
        new_population = []

        for i in range(self.population_size):
            current_wheel_probability = 0.
            random_probability = random.random()

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
        Randomly distributes the population into pairs and applies arithmetic crossover with a given probability.
        """
        random_indexes = np.random.permutation(self.population_size)

        for i in range(0, len(random_indexes) - 1, 2):
            if np.random.rand() < self.crossover_probability:
                # Select parents
                parent1 = self.population[random_indexes[i]]
                parent2 = self.population[random_indexes[i + 1]]

                # Generate offspring using arithmetic crossover
                alpha = np.random.rand()  # Weight for linear combination
                offspring1 = alpha * parent1 + (1 - alpha) * parent2
                offspring2 = alpha * parent2 + (1 - alpha) * parent1

                # Replace parents with offspring
                self.population[random_indexes[i]] = offspring1
                self.population[random_indexes[i + 1]] = offspring2

    def mutate(self):
        """
        Applies arithmetic mutation to the population with a given probability.
        """
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_probability:
                # Select a chromosome to mutate
                chromosome = self.population[i]

                # Apply arithmetic mutation
                alpha = np.random.rand()  # Weight for mutation
                mutation_vector = np.random.uniform(-0.1, 0.1, size=chromosome.shape)  # Small random changes
                self.population[i] = (1 - alpha) * chromosome + alpha * (chromosome + mutation_vector)

    def draw_errors(self):
        figure = plt.subplot()
        iters = np.array(range(1, self.max_generations + 1))
        plt.plot(iters, self.min_errors, label=fr"Train best loss (ED)")
        plt.plot(iters, self.mean_errors, label=fr"Train mean (ED)")
        plt.xlabel("Generation")
        plt.ylabel("Euclidian Distance")

        plt.title(
            fr"P_c: {self.crossover_probability}; P_m: {self.mutation_probability}, founded solution: E_f={self.best_solution[0]:.2f} * L ^ {self.best_solution[1]:.2f}")
        plt.legend()

        plt.show()
