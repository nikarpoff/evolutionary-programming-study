import genalg

POPULATION_SIZE = 200

CROSSING_OVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.01
MAX_GENERATIONS = 100
FILENAME = "berlin52.tsp"

if __name__ == '__main__':
    genetic_optimizer = genalg.GeneticTSPSolver(FILENAME, MAX_GENERATIONS, POPULATION_SIZE, CROSSING_OVER_PROBABILITY,
                                                MUTATION_PROBABILITY)

    genetic_optimizer.start()
