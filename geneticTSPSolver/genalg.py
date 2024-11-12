import matplotlib.pyplot as plt
import numpy as np

from time import time
import random


def count_euclidean_distance(city1, city2):
    """
    Calculates the Euclidean distance between two coordinate points.
    :param city1: first (x, y) coordinates
    :param city2: second (x, y) coordinates
    :return: float number - distance between two points
    """
    return np.linalg.norm(city1 - city2)


def check_nlr_for_cycle(neighbor_list):
    """
    Checks if the is valid neighbor representation list (closed, without cycles)
    :param neighbor_list: list of neighbors (chromosome)
    :return: True if it is valid, False otherwise
    """
    # return True
    n = len(neighbor_list)
    visited = [False] * n  # All visited cities.
    current_city = 0  # Start from city 0.

    for _ in range(n):
        if visited[current_city]:
            # If current city is visited then we got cycle.
            return False
        visited[current_city] = True
        next_city = neighbor_list[current_city] - 1

        # Go to next city.
        current_city = next_city

    # After visiting n cities check if we are in the start city.
    return current_city == 0


def convert_neighbor_list_to_route(neighbor_list, start_city=1):
    """
    Converts a neighbor list representation into an ordered route.
    :param neighbor_list: list of neighbors (neighbor list representation)
    :param start_city: city to start the route (1-based index)
    :return: list of cities in the order of the route
    """
    n = len(neighbor_list)
    route = [start_city]
    current_city = start_city

    for _ in range(n - 1):
        next_city = neighbor_list[current_city - 1]
        route.append(next_city)
        current_city = next_city

    return route


def format_route(route):
    """
    Formats the given route as a string with cities connected by dashes.
    :param route: list of cities
    :return: string representing the route
    """
    return '-'.join(map(str, route))


class GeneticTSPSolver:
    """
    Solves task of finding maximum of target function (fitness function) using genetic algorithm.
    :param tsp_filename: file with tsp initial data
    :param population_size: size of the population
    :param max_generations: number of max available generations. For a situation when it is impossible to find a satisfying optimum
    :param crossover_probability: probability of a crossover
    :param mutation_probability: probability of a mutation
    """

    def __init__(self, tsp_filename: str, max_generations: int, population_size: int,
                 crossover_probability: float, mutation_probability: float):
        self.crossover_probability = crossover_probability
        self.max_generations = max_generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.cities = []
        self.dimensions = 0

        # Read from file coordinates, cities and number of cities (dimensions).
        self.coords = self.read_tsp_file(tsp_filename)

        # Distances matrix.
        self.distances = np.zeros(shape=(self.dimensions, self.dimensions))
        self.calculate_distances()

        self.population = []
        self.total_distances = []

        # Initialize population with random shuffled cities arrays
        for i in range(population_size):
            is_invalid_way = True

            while is_invalid_way:
                chromosome = self.cities.copy()
                random.shuffle(chromosome)

                # Check is there random generated way valid.
                is_invalid_way = not check_nlr_for_cycle(chromosome)

                if not is_invalid_way:
                    self.population.append(chromosome)
                    self.total_distances.append(0.)

        # array = [49, 7, 18, 6, 24, 15, 42, 9, 10, 43, 52, 25, 47, 13, 5, 29, 3, 31, 41, 23, 17, 1, 30, 48, 4, 27, 28,
        #          12, 50, 2, 22, 45, 51, 44, 34, 35, 40, 37, 36, 39, 8, 21, 33, 46, 19, 16, 26, 38, 32, 20, 11, 14]
        #
        # self.population = [array, array, array, array]

    def read_tsp_file(self, filename) -> np.array:
        """
        Reads tsp file and save data to self cities, coords and dimensions (number of dimensions)
        :param filename: name of the tsp file
        """
        with open(filename, 'r') as file:
            lines = file.readlines()

        start_read_coords = False

        coords = []

        for line in lines:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                start_read_coords = True
                continue

            if line == "EOF":
                break

            if start_read_coords:
                parts = line.split()
                self.cities.append(int(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

        self.dimensions = len(self.cities)
        return np.array(coords)


    def calculate_distances(self):
        """Creates Matrix of distances between cities"""
        for i in range(self.dimensions):
            for j in range(self.dimensions):
                self.distances[i][j] = count_euclidean_distance(self.coords[i], self.coords[j])

    def start(self):
        generation = 1  # generations counter

        start_time = time()

        while generation < self.max_generations:
            # 1. Evaluate current distances.
            self.evaluate()

            new_population = self.population.copy()

            for i in range(0, self.population_size, 2):
                # 2. Reproduction.
                element1 = self.reproduction()
                element2 = self.reproduction()

                # 3. Crossing.
                if random.random() < self.crossover_probability:
                    element1 = self.crossover(element1, element2)
                    element2 = self.crossover(element2, element1)

                new_population[i] = element1
                new_population[i + 1] = element2

            for i in range(len(new_population)):
                self.population[i] = new_population[i]

            # 4. Mutate.
            self.mutate()

            print(f'Generation [{generation}/{self.max_generations}]\t\t' +
                  f'Min distance = {min(self.total_distances):.3f}\t\t')

            generation += 1

        end_time = time()

        print('\n', '=' * 100)

        best_index = self.total_distances.index(min(self.total_distances))
        best_solution = self.population[best_index]
        route = convert_neighbor_list_to_route(best_solution)
        formatted_route = format_route(route)

        print(f'Required time: {end_time - start_time:.2f}s. Found answer: {min(self.total_distances):4f}.\n',
              f'Best solution: {formatted_route}\n',
              f'({best_solution})',
              f'Pc = {self.crossover_probability}, Pm = {self.mutation_probability}')

        self.plot_route(best_solution)

    def evaluate(self):
        """
        Evaluates current total distances for current population.
        """
        for i in range(len(self.population)):
            current_total_distance = 0.

            # Count total distance for every chromosome in population.
            for j in range(self.dimensions):
                # Use neighbour representation.
                current_city = self.population[i][j]
                prev_city = j + 1

                distance = self.distances[current_city - 1, prev_city - 1]
                current_total_distance += distance

            self.total_distances[i] = current_total_distance

    def reproduction(self, k=3):
        """
        Tournament based reproduction algorithm.
        """
        selected = random.sample(list(zip(self.population, self.total_distances)), k)
        return min(selected, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        """
        Heuristic crossover between two parents.
        :param parent1: first parent (chromosome)
        :param parent2: second parent (chromosome)
        :return: child (chromosome)
        """
        child = [-1] * self.dimensions  # Empty child chromosome
        used_cities = set()  # Set of cities that have already been added to the child

        # Start from a random city from parent1
        current_city = random.randint(1, self.dimensions)
        start_city = current_city
        used_cities.add(start_city)

        for i in range(1, self.dimensions+1):
            # Find the next city from parent1 or parent2 based on the shortest distance
            next_city_p1 = parent1[current_city - 1]
            next_city_p2 = parent2[current_city - 1]

            # Choose the city with the shortest distance
            if next_city_p1 not in used_cities and next_city_p2 not in used_cities:
                distance_p1 = self.distances[current_city - 1][next_city_p1 - 1]
                distance_p2 = self.distances[current_city - 1][next_city_p2 - 1]

                if distance_p1 < distance_p2:
                    next_city = next_city_p1
                else:
                    next_city = next_city_p2
            elif next_city_p1 in used_cities and next_city_p2 in used_cities:
                # If no valid next city, choose a random unvisited city
                remaining_cities = list(set(range(1, self.dimensions + 1)) - used_cities)
                if len(remaining_cities) > 0:
                    next_city = random.choice(remaining_cities)
                else:
                    # All cities were visited -> next city = start city
                    next_city = start_city
            elif next_city_p2 in used_cities:
                next_city = next_city_p1
            else:
                next_city = next_city_p2

            # Add the chosen city to the child
            child[current_city - 1] = next_city
            used_cities.add(next_city)

            current_city = next_city

        return child

    def select(self):
        """Selects population_size best chromosomes from current_population into population."""
        while len(self.population) > self.population_size:
            worse_chromosome_index = self.total_distances.index(max(self.total_distances))
            self.population.pop(worse_chromosome_index)
            self.total_distances.pop(worse_chromosome_index)

    def mutate(self):
        """
        Inversion mutation: inverts the order of a random subsequence of cities in the route.
        """
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                # Randomly select two cities to invert the segment between them
                city1, city2 = sorted(random.sample(range(self.dimensions), 2))

                # Invert the subsequence between city1 and city2
                self.population[i][city1:city2 + 1] = reversed(self.population[i][city1:city2 + 1])

    def plot_route(self, best_route):
        """
        Plots the best found TSP route.
        """
        plt.figure(figsize=(8, 8))

        plt.scatter(self.coords[:, 0], self.coords[:, 1], color='red')

        for i in range(len(best_route)):
            plt.plot([self.coords[i, 0]-1, self.coords[best_route[i]-1, 0]],
                     [self.coords[i, 1]-1, self.coords[best_route[i]-1, 1]], 'b')

        for i, city in enumerate(best_route):
            plt.text(self.coords[city - 1][0], self.coords[city - 1][1], str(city), fontsize=12)

        plt.title('Best TSP Route')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()
