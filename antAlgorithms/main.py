import numpy as np
import genalg

POPULATION_SIZE = 200
ALPHA = 0.95
RHO = 0.8
MAX_GENERATIONS = 50

graph_hamilton = np.array([
    # 0  1  2  3  4  5  6
    [0, 1, 0, 1, 0, 0, 1],  # 0
    [0, 0, 1, 0, 1, 0, 0],  # 1
    [0, 0, 0, 1, 0, 1, 0],  # 2
    [0, 1, 0, 0, 1, 0, 0],  # 3
    [0, 1, 0, 1, 0, 0, 0],  # 4
    [0, 0, 0, 1, 1, 0, 1],  # 5
    [0, 0, 0, 0, 0, 0, 0],  # 6
])


def parse_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coords = []
    start = False
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            start = True
            continue
        if start:
            if line.startswith("EOF"):
                break
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2])))  # (x, y)
    return coords


def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def generate_distance_matrix(coords):
    n = len(coords)
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = calculate_distance(coords[i], coords[j])
    return graph


FILENAME = "berlin52.tsp"
coords = parse_tsp_file(FILENAME)
graph_tsp = generate_distance_matrix(coords)


if __name__ == '__main__':
    # genetic_optimizer = genalg.AntColonyTask(POPULATION_SIZE, MAX_GENERATIONS, ALPHA, RHO, graph_hamilton)
    # genetic_optimizer.start_hamilton()

    genetic_optimizer = genalg.AntColonyTask(POPULATION_SIZE, MAX_GENERATIONS, ALPHA, RHO, graph_tsp, coords)
    genetic_optimizer.start_tsp()

