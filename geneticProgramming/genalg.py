import matplotlib.pyplot as plt

import random
import numpy as np
from time import time

from tree import TreeGenerator, collect_nodes, replace_node


class GeneticProgrammingAlgorithm:
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

    def __init__(self, target_function, max_generations: int, x: np.ndarray, y: np.ndarray,
                 terminals: list, unary_functions: dict, binary_functions: dict, population_size: int,
                 max_deep_size: int, crossover_probability: float, mutation_probability: float):
        self.target_function = target_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.unary_functions = unary_functions
        self.binary_functions = binary_functions
        self.functions = list(unary_functions.keys()) + list(binary_functions.keys())
        self.terminals = terminals
        self.max_deep_size = max_deep_size

        self.x = x
        self.y_true = y

        # Every chromosome is tree of functions and terminals
        self.population = []

        self.tree_generator = TreeGenerator(terminals, unary_functions, binary_functions, self.functions)

        for i in range(population_size):
            tree_depth = random.randint(2, max_deep_size)
            tree = self.tree_generator.generate(tree_depth)
            self.population.append(tree)

        self.losses_values = np.ones(population_size, float)

    def start(self):
        generation = 1  # generations counter

        start_time = time()

        while generation < self.max_generations:
            # 1. Evaluate fitness.
            self.evaluate()

            best_solution_idx = np.where(self.losses_values == min(self.losses_values))[0][0]
            best_solution = self.population[best_solution_idx]
            print(best_solution.build())

            # 2. Reproduction.
            self.reproduction()

            # 3. Crossing.
            self.crossover()

            # 4. Mutate.
            self.mutate()

            print(f'Generation [{generation}/{self.max_generations}]\t\t' +
                  f'\t\tMin fitness = {min(self.losses_values):.3f}\t\t')

            generation += 1

        end_time = time()

        best_solution_idx = np.where(self.losses_values == min(self.losses_values))[0][0]
        best_solution = self.population[best_solution_idx]

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Result mae: {min(self.losses_values):4f}. ',
              f'Required generations: {generation}. Pc = {self.crossover_probability}, ',
              f'Pm = {self.mutation_probability}\nFound solution: {best_solution.build()}')

    def evaluate(self):
        """
        Evaluates the fitness of the population.
        """
        # Count fitness of every chromosome in population.
        for i in range(self.population_size):
            y = self.population[i].count(self.x)
            mean_absolute_error = np.mean(np.abs(self.y_true - y))
            self.losses_values[i] = mean_absolute_error

    def reproduction(self, k=3):
        """
        Tournament based reproduction algorithm.
        """
        selected = random.sample(list(zip(self.population, self.losses_values)), k)
        return min(selected, key=lambda x: x[1])[0]

    def crossover(self):
        """
        Randomly distributes population on pairs and makes crossing (with probability).
        """
        random_indexes = np.random.permutation(self.population_size)

        for i in range(0, len(random_indexes) - 1, 2):
            # With probability.
            if random.random() < self.crossover_probability:
                # Choose two nodes from paired trees.
                parent1, parent2 = self.population[random_indexes[i]], self.population[random_indexes[i + 1]]

                nodes1, nodes2 = collect_nodes(parent1), collect_nodes(parent2)

                node1, node2 = random.choice(nodes1), random.choice(nodes2)

                # We can swap nodes with the same type
                tries = 0
                while not isinstance(node1, type(node2)) and tries < 10:
                    node2 = random.choice(nodes2)
                    tries += 1

                if tries != 10:
                    node1.label = node2.label

    def mutate(self):
        """
        Subtree mutation for population with probability
        """
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                # Choose two nodes from paired trees.
                tree_root = self.population[i]
                tree_depth = tree_root.get_depth()

                listed_nodes = collect_nodes(tree_root)
                random_node = random.choice(listed_nodes)

                random_node_depth = random_node.get_depth()
                new_node = self.tree_generator.generate(tree_depth - random_node_depth)

                replace_node(tree_root, random_node, new_node)

    def draw_plot(self, generation):
        dots_n = 100

        x = np.linspace(self.left_edge, self.right_edge, dots_n)
        Z = np.zeros(shape=(dots_n, dots_n))

        for i in range(dots_n):
            for j in range(dots_n):
                Z[i][j] = self.target_function(np.array([x[i], x[j]]))

        # Plot the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, x)
        ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.7)

        x_dots = self.population[:, 0]
        y_dots = self.population[:, 1]
        z_dots = self.losses_values

        ax.scatter3D(x_dots, y_dots, z_dots, color='green', marker='o', s=50, edgecolor='black')

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"Generation {generation}. Population size is {self.population_size}.\n" +
                     f"Pc -> {self.crossover_probability}. Pm -> {self.mutation_probability}. " +
                     f"Min fitness = {min(self.losses_values):.4f}")

        plt.show()
