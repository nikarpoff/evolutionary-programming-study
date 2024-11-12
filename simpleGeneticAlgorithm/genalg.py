import matplotlib.pyplot as plt
import numpy as np

from time import time
from random import randint, random

UNSIGNED_INT_BITS_NUMBER = 16


def decode_number(segment_number: int, left_interval, right_interval) -> float:
    """
    Decode an int number specified segment on [left_interval, right_interval] into float number from this interval.
    """
    # x = b - x' * (a - b) / segments_number
    return left_interval + segment_number * (right_interval - left_interval) / (2 ** UNSIGNED_INT_BITS_NUMBER - 1)


def swap_bits(first_chromosome, second_chromosome, k):
    """
    Swaps bits between two chromosomes from k bit's position.
    :param first_chromosome: first number to be swapped
    :param second_chromosome: second number to be swapped
    :param k: bit's position to start swap
    :return: inverted number
    """
    # Create strings with a binary representations of numbers.
    first_chromosome_original = bin(first_chromosome)[2:].zfill(16)
    second_chromosome_original = bin(second_chromosome)[2:].zfill(16)

    # Swap bits from k
    swapped_first_chromosome = f'0b{first_chromosome_original[:k]}{second_chromosome_original[k:]}'
    swapped_second_chromosome = f'0b{second_chromosome_original[:k]}{first_chromosome_original[k:]}'

    return int(swapped_first_chromosome, 2), int(swapped_second_chromosome, 2)


def invert_bit(chromosome, k):
    """
    Inverts one bit from k position on 16-bits number.
    :param chromosome: number to be inverted
    :param k: bit's position to invert
    :return: inverted number
    """
    # Create a string with a binary representation of a number.
    original = bin(chromosome)[2:].zfill(16)
    inverted = original[:k - 1]

    if original[k - 1] == '0':
        inverted += '1'
    else:
        inverted += '0'

    inverted += original[k:]

    return int(f'0b{inverted}', 2)


class SimpleGeneticAlgorithm:
    """
    Solves task of finding maximum of target function (fitness function) using genetic algorithm.
    :param fitness_function: target function
    :param population_size: size of the population
    :param max_generations: number of max available generations. For a situation when it is impossible to find a satisfying optimum
    :param crossover_probability: probability of a crossover
    :param mutation_probability: probability of a mutation
    :param epsilon: stop algorithm when  mean fitness function is less than or equal to this value
    """

    def __init__(self, fitness_function, population_size, max_generations,
                 crossover_probability, mutation_probability, left_edge, right_edge, epsilon=0.05):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.epsilon = epsilon

        self.population = []
        self.fitness_values = np.zeros(population_size, float)

        for i in range(population_size):
            # Decode with 16-bits number.
            self.population.append(randint(0, 65535))

    def start(self):
        generation = 1  # generations counter
        last_fitness_mean = 0.  # mean fitness
        mean_fitness_delta = 1.  # delta between current and previous mean fitness function value

        plt.ion()
        start_time = time()

        while generation < self.max_generations and abs(mean_fitness_delta) > self.epsilon:
            # 1. Evaluate fitness.
            self.evaluate()

            # 2. Reproduction.
            self.reproduction()

            # 3. Crossing.
            self.crossover()

            # 4. Mutate.
            self.mutate()

            mean_fitness_delta = self.fitness_values.mean() - last_fitness_mean
            last_fitness_mean = self.fitness_values.mean()

            print(f'Generation [{generation}/{self.max_generations}]\t\t' +
                  f'Fitness delta = {mean_fitness_delta:.3f}\t\tMax fitness = {max(self.fitness_values):.3f}\t\t' +
                  f'Mean: {(sum(self.fitness_values) / self.population_size):.3f}\n')

            self.draw_plot(generation)

            generation += 1

        end_time = time()
        plt.show(block=True)

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Found answer: {max(self.fitness_values):4f}. ',
              f'Required generations: {generation}. Fitness mean delta: {abs(mean_fitness_delta):3f}')

    def evaluate(self):
        """
        Evaluates the fitness of the population.
        """
        for i in range(self.population_size):
            # Count fitness of every chromosome in population
            self.fitness_values[i] = self.fitness_function(decode_number(self.population[i],
                                                                         self.left_edge,
                                                                         self.right_edge)
                                                           )

    def reproduction(self):
        """
        Roulette based reproduction algorithm.
        """
        fitness_copy = []

        # Normalize to avoid problems with negative numbers.
        min_fitness = min(self.fitness_values)
        for fitness in self.fitness_values:
            if min_fitness < 0:
                fitness_copy.append(fitness + abs(min_fitness) + 0.0000000001)
            else:
                fitness_copy.append(fitness)

        total_fitness = sum(fitness_copy)
        probabilities = [fitness / total_fitness for fitness in fitness_copy]
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
            if i != new_population.__len__() - 1:
                new_population.append(self.population[-1])

        # Update population.
        for i in range(self.population_size):
            self.population[i] = new_population[i]

    def crossover(self):
        """
        Randomly distributes population on pairs and makes crossing (with probability).
        """
        free_chromosomes = self.population.copy()
        pairs = []

        # While there is free chromosomes to make pair.
        while len(free_chromosomes) > 1:
            pair = []

            # Choose two random elements.
            for i in range(2):
                rand_index = randint(0, len(free_chromosomes) - 1)
                pair.append(free_chromosomes.pop(rand_index))

            pairs.append(pair)

        for pair in pairs:
            # With probability.
            if random() < self.crossover_probability:
                cross_dot = randint(1, UNSIGNED_INT_BITS_NUMBER - 1)

                first_index = self.population.index(pair.pop())
                second_index = self.population.index(pair.pop())

                first_swapped, second_swapped = swap_bits(self.population[first_index],
                                                          self.population[second_index],
                                                          cross_dot)

                self.population[first_index], self.population[second_index] = first_swapped, second_swapped

    def mutate(self):
        for i in range(self.population_size):
            if random() < self.mutation_probability:
                rand_index = randint(1, UNSIGNED_INT_BITS_NUMBER)
                self.population[i] = invert_bit(self.population[i], rand_index)

    def draw_plot(self, generation):
        plt.clf()

        x = [decode_number(i, self.left_edge, self.right_edge) for i in range(0, 65536, 10)]
        y = [self.fitness_function(value) for value in x]
        plt.plot(x, y, label='y = (x-1) cos(3x - 15)')

        decoded_dots = []
        decoded_dots_y = []
        for dot in self.population:
            decoded_dots.append(decode_number(dot, self.left_edge, self.right_edge))
            decoded_dots_y.append(self.fitness_function(decode_number(dot, self.left_edge, self.right_edge)))

        plt.scatter(decoded_dots, decoded_dots_y, color='red')

        plt.xlim(self.left_edge, self.right_edge)
        plt.ylim(-12, 12)
        plt.title(f"Generation {generation}. Population size is {self.population_size}.\n" +
                  f"Pc -> {self.crossover_probability}. Pm -> {self.mutation_probability}. " +
                  f"Max fitness = {max(self.fitness_values):.4f}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.draw()
        plt.pause(0.1)
