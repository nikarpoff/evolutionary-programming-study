import numpy as np
import genalg

POPULATION_SIZE = 100
OFFSPRING_SIZE = 20
CROSSING_OVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.01
MAX_GENERATIONS = 50
EPSILON = 0.
N = 3

LEFT_EDGE, RIGHT_EDGE = -5.12, 5.12


# Function is Rastrigin's function 6
def fitness_function(chromosome):
    return 10 * len(chromosome) + np.sum(np.power(chromosome, 2) - 10 * np.cos(2 * np.pi * chromosome))


if __name__ == '__main__':
    genetic_optimizer = genalg.EvolutionaryStrategy(fitness_function, N, POPULATION_SIZE, MAX_GENERATIONS,
                                                    OFFSPRING_SIZE, CROSSING_OVER_PROBABILITY, MUTATION_PROBABILITY,
                                                    LEFT_EDGE, RIGHT_EDGE)

    genetic_optimizer.start()
