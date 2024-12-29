import numpy as np
import genalg

POPULATION_SIZE = 100
MAX_GENERATIONS = 50
C1 = 2.
C2 = 1.5
W = 0.9
N = 5

LEFT_EDGE, RIGHT_EDGE = -5.12, 5.12


# Function is Rastrigin's function 6
def fitness_function(x):
    return 10 * len(x) + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))


if __name__ == '__main__':
    pso_optimizer = genalg.AlgorithmPSO(fitness_function, N, POPULATION_SIZE, MAX_GENERATIONS,
                                        LEFT_EDGE, RIGHT_EDGE, C1, C2, W)

    pso_optimizer.start()
