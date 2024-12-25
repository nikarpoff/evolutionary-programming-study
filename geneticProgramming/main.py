import numpy as np
from sklearn import preprocessing

from genalg import GeneticProgrammingAlgorithm


def target_function(x: np.ndarray) -> float:
    """
    Target function is sum(-x*sin(sqrt(abs(x))))
    :param x: vector x
    :return: value of function
    """
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    # return np.sum(x)


left = -500.
right = 500.
n = 2
size = 1000

# Generate data.
np.random.seed(42)
X = np.random.uniform(low=left, high=right, size=(size, n))

scaler = preprocessing.MinMaxScaler()
# X = scaler.fit_transform(X, {'feature_range': (-5, 5)})

y_true = np.apply_along_axis(target_function, axis=1, arr=X)

# Define terminal set.
# terminals = list(f'x{i + 1}' for i in range(n)) + (list(str(i) for i in range(-3, 3)))
terminals = list(f'x{i + 1}' for i in range(n)) + ['0']


# Define functions set.
unary_functions = {
    'abs': lambda x: np.abs(x),
    'sin': lambda x: np.sin(x),
    'sqrt': lambda x: np.sqrt(np.clip(x, a_min=0, a_max=1e10)),
    # 'cos': lambda x: np.cos(x),
    # 'exp': lambda x: np.exp(x)
}

binary_functions = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    # '/': lambda a, b: a / (b + 1e-6),
    # '**': lambda a, b: np.power(a, b)
}

MAX_GENERATIONS = 1000
POPULATION_SIZE = 1000
MAX_DEEP_SIZE = 20
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2

if __name__ == '__main__':
    solver = GeneticProgrammingAlgorithm(target_function, MAX_GENERATIONS, X, y_true, terminals,
                                         unary_functions, binary_functions, POPULATION_SIZE, MAX_DEEP_SIZE,
                                         CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)

    solver.start()
