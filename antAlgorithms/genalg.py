import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from time import time
import random


class AntColonyTask:
    """
    Solves task of finding Hamilton path in oriented graph.
    :param population_size: size of the population
    :param max_iterations: number of max available iterations
    :param alpha: influence of pheromone
    :param rho: coefficient of pheromone evaporation
    """

    def __init__(self, population_size: int, max_iterations: int, alpha: float, rho: float,
                 distance_matrix, coords=None):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.rho = rho
        self.distance_matrix = distance_matrix
        self.coords = coords

        self.total_nodes = distance_matrix.shape[0]
        self.pheromone_matrix = np.ones_like(distance_matrix) / len(distance_matrix)

    def calculate_transition_probabilities(self, current_node, unvisited):
        probabilities = []
        total_tau = 0.

        for next_node in unvisited:
            if self.distance_matrix[current_node][next_node] > 0:
                total_tau += (self.pheromone_matrix[current_node][next_node] ** self.alpha)

        for next_node in unvisited:
            probabilities.append(self.pheromone_matrix[current_node][next_node] ** self.alpha
                                 / total_tau if self.distance_matrix[current_node][next_node] > 0 else 0)

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum() if probabilities.sum() > 0 else probabilities

    def calculate_transition_probabilities_tsp(self, current_node, unvisited):
        pheromones = self.pheromone_matrix[current_node, list(unvisited)]
        distances = self.distance_matrix[current_node, list(unvisited)]

        attractiveness = pheromones ** self.alpha / distances
        return attractiveness / attractiveness.sum()

    def start_hamilton(self):
        iteration = 1  # iterations counter

        best_path = None
        best_unvisited_total = float('inf')
        best_visited_total = 0

        start_time = time()

        for iteration in range(self.max_iterations):
            paths = []
            path_lengths = []

            for ant in range(self.population_size):
                current_node = 0
                path = [current_node]
                unvisited = set(range(self.total_nodes)) - {current_node}

                while current_node != self.total_nodes - 1:
                    probabilities = self.calculate_transition_probabilities(current_node, unvisited)

                    if np.sum(probabilities) != 1.:
                        break

                    next_node = random.choices(list(unvisited), weights=probabilities, k=1)[0]
                    unvisited.remove(next_node)
                    path.append(next_node)
                    current_node = next_node

                unvisited_total = len(unvisited)
                visited_total = self.total_nodes - unvisited_total
                path_lengths.append(visited_total)
                paths.append(path)

                # Обновляем лучший путь
                if unvisited_total < best_unvisited_total:
                    best_unvisited_total = unvisited_total
                    best_visited_total = visited_total
                    best_path = path

            # Испарение феромона
            self.pheromone_matrix *= (1 - self.rho)

            # Обновление феромона
            for path, visited_total in zip(paths, path_lengths):
                for i in range(len(path) - 1):
                    self.pheromone_matrix[path[i]][path[i + 1]] += 1 / (visited_total + 1)

            print(f'Iteration [{iteration + 1}/{self.max_iterations}]\t\t' +
                  f'Visited nodes = {best_visited_total}\t\t total pheromone = {np.sum(self.pheromone_matrix):.3f}\n')

            self.draw_graph(iteration, best_visited_total, best_path)

            iteration += 1

        end_time = time()

        print('\n', '=' * 100)
        print(f'Required time: {end_time - start_time:.2f}s. Found answer: {best_path}. ',
              f'Required generations: {iteration}. Total pheromone: {np.sum(self.pheromone_matrix):.3f}')

    def calculate_path_length(self, path):
        return sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def two_opt(self, path):
        best_path = path
        best_length = self.calculate_path_length(path)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    new_path = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                    new_length = self.calculate_path_length(new_path)
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
                        improved = True
        return best_path

    def start_tsp(self):
        best_path = None
        best_path_length = float('inf')
        start_time = time()

        for iteration in range(self.max_iterations):
            paths = []
            path_lengths = []

            for ant in range(self.population_size):
                current_node = random.randint(0, self.total_nodes - 1)
                path = [current_node]
                unvisited = set(range(self.total_nodes)) - {current_node}

                while unvisited:
                    probabilities = self.calculate_transition_probabilities_tsp(current_node, unvisited)
                    next_node = random.choices(list(unvisited), weights=probabilities, k=1)[0]
                    unvisited.remove(next_node)
                    path.append(next_node)
                    current_node = next_node

                path.append(path[0])  # Замыкаем путь

                path = self.two_opt(path)  # Локальная оптимизация
                path_length = self.calculate_path_length(path)

                paths.append(path)
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path_length = path_length
                    best_path = path

            # Испарение и обновление феромонов
            self.pheromone_matrix *= (1 - self.rho)

            for path, length in zip(paths, path_lengths):
                for i in range(len(path) - 1):
                    self.pheromone_matrix[path[i]][path[i + 1]] += 1 / length

            print(f"Iteration {iteration + 1}/{self.max_iterations}: Best path length = {best_path_length}")
            self.draw_with_coords(iteration, best_path_length, best_path)

        end_time = time()
        print(f"Best path length: {best_path_length}, Time: {end_time - start_time:.2f}s")

    def draw_with_coords(self, iteration, best_len=float('inf'), best_path=None):
        G = nx.Graph()
        for i, (x, y) in enumerate(self.coords):
            G.add_node(i, pos=(x, y))

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')

        if best_path:
            path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

        plt.title(f"Iteration {iteration}. Length: {best_len:.2f}")
        plt.show()

    def draw_graph(self, iteration, best_len=float('inf'), best_path=None):
        G = nx.DiGraph()
        for i in range(len(self.distance_matrix)):
            for j in range(len(self.distance_matrix)):
                if self.distance_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=self.distance_matrix[i][j])

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{w['weight']}" for i, j, w in G.edges(data=True)})

        if best_path:
            path_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

        plt.title(f"Iteration {iteration}. Length: {best_len}")
        plt.show()
