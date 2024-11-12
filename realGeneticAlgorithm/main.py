import numpy as np
import genalg

POPULATION_SIZE = 100
CROSSING_OVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 50
EPSILON = 0.
N = 2

LEFT_EDGE, RIGHT_EDGE = -5.12, 5.12


# Function is Rastrigin's function 6
def fitness_function(chromosome):
    return 10 * len(chromosome) + np.sum(np.power(chromosome, 2) - 10 * np.cos(2 * np.pi * chromosome))


if __name__ == '__main__':
    genetic_optimizer = genalg.RealGeneticAlgorithm(fitness_function, N, POPULATION_SIZE, MAX_GENERATIONS,
                                                    CROSSING_OVER_PROBABILITY, MUTATION_PROBABILITY,
                                                    LEFT_EDGE, RIGHT_EDGE, EPSILON)

    genetic_optimizer.start()
