from math import cos
import genalg

POPULATION_SIZE = 100
CROSSING_OVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.001
MAX_GENERATIONS = 100
EPSILON = 0.0005

LEFT_EDGE, RIGHT_EDGE = -10, 10


# Function is (x-1) cos(3x - 15)
def fitness_function(chromosome):
    return (chromosome - 1) * cos(3 * chromosome - 15)


if __name__ == '__main__':
    genetic_optimizer = genalg.SimpleGeneticAlgorithm(fitness_function, POPULATION_SIZE, MAX_GENERATIONS,
                                                      CROSSING_OVER_PROBABILITY, MUTATION_PROBABILITY,
                                                      LEFT_EDGE, RIGHT_EDGE, EPSILON)

    genetic_optimizer.start()
