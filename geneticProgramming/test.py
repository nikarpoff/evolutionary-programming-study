import numpy as np
import random
import math
from copy import deepcopy
from sklearn import preprocessing

# Константы
MAX_DEPTH = 5
POPULATION_SIZE = 10
TOURNAMENT_SIZE = 3
MUTATION_PROBABILITY = 0.2
CROSSOVER_PROBABILITY = 0.7

# Функции и терминалы для дерева
TERMINALS = [f'x{i}' for i in range(1, 11)] + [random.uniform(-10, 10) for _ in range(5)]
FUNCTIONS = ['+', '-', '*', '/', 'cos', 'exp']


# Функция для вычисления целевой функции
def target_function(X):
    return sum(math.cos(x) + 3 for x in X)


# Узел дерева
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def is_function(self):
        return self.value in FUNCTIONS

    def is_terminal(self):
        return self.value not in FUNCTIONS


# Генерация случайного дерева
def generate_tree(max_depth, depth=0):
    if depth >= max_depth or (depth > 1 and random.random() < 0.5):  # Лист или константа
        return Node(random.choice(TERMINALS))
    else:  # Функция
        func = random.choice(FUNCTIONS)
        if func in ['+', '-', '*', '/']:
            return Node(func, [generate_tree(max_depth, depth + 1), generate_tree(max_depth, depth + 1)])
        elif func in ['cos', 'exp']:
            return Node(func, [generate_tree(max_depth, depth + 1)])


# Инициализация популяции
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE // 2):
        depth = random.choice([2, 3, 5])
        population.append(generate_tree(depth))
    for _ in range(POPULATION_SIZE // 2):
        depth = random.randint(1, MAX_DEPTH)
        population.append(generate_tree(depth))
    return population


# Оценка выражения дерева
def evaluate_tree(node, variables):
    if node.is_terminal():
        if isinstance(node.value, str) and node.value.startswith('x'):
            return variables[int(node.value[1:]) - 1]
        return node.value
    elif node.value == '+':
        return evaluate_tree(node.children[0], variables) + evaluate_tree(node.children[1], variables)
    elif node.value == '-':
        return evaluate_tree(node.children[0], variables) - evaluate_tree(node.children[1], variables)
    elif node.value == '*':
        return evaluate_tree(node.children[0], variables) * evaluate_tree(node.children[1], variables)
    elif node.value == '/':
        denominator = evaluate_tree(node.children[1], variables)
        return evaluate_tree(node.children[0], variables) / denominator if denominator != 0 else 1
    elif node.value == 'cos':
        return math.cos(evaluate_tree(node.children[0], variables))
    elif node.value == 'exp':
        return math.exp(evaluate_tree(node.children[0], variables))


# Фитнесс-функция
def fitness_function(tree, X, y_true):
    y_pred = [evaluate_tree(tree, x) for x in X]
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


# Получение всех узлов дерева
def get_nodes(node):
    nodes = [node]
    if node.children:
        for child in node.children:
            nodes.extend(get_nodes(child))
    return nodes


# Поиск совместимого узла для обмена
def find_compatible_node(tree, target_node):
    candidate_nodes = get_nodes(tree)
    compatible_nodes = [node for node in candidate_nodes if node.is_function() == target_node.is_function()]
    return random.choice(compatible_nodes) if compatible_nodes else None


# Обмен узлов
def swap_nodes(tree1, tree2, node1, node2):
    # Находим ссылки на родительские узлы
    def find_parent(root, target):
        if root.children:
            for i, child in enumerate(root.children):
                if child == target:
                    return root, i
                result = find_parent(child, target)
                if result:
                    return result
        return None

    # Находим родителя для каждого узла
    result1 = find_parent(tree1, node1)
    result2 = find_parent(tree2, node2)

    if result1 and result2:
        # Меняем узлы местами
        parent1, index1 = result1
        parent2, index2 = result2

        parent1.children[index1], parent2.children[index2] = parent2.children[index2], parent1.children[index1]


# Оператор кроссинговера
def crossover(parent1, parent2):
    node1 = random.choice(get_nodes(parent1))
    node2 = find_compatible_node(parent2, node1)
    if node2:
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        swap_nodes(child1, child2, node1, node2)
        return child1, child2
    return deepcopy(parent1), deepcopy(parent2)


# Оператор мутации
def mutation(tree):
    node = random.choice(get_nodes(tree))
    if node.is_function():
        node.children = [generate_tree(random.randint(1, MAX_DEPTH)) for _ in range(len(node.children))]
    return tree


# Функция для вывода дерева в виде математического выражения
def tree_to_expression(node):
    if node.is_terminal():
        # Если это терминал (константа или переменная), возвращаем его значение как строку
        return str(node.value)
    elif node.value in ['+', '-', '*', '/']:
        # Бинарные операторы - оборачиваем операнды в скобки для корректного отображения
        left_expr = tree_to_expression(node.children[0])
        right_expr = tree_to_expression(node.children[1])
        return f"({left_expr} {node.value} {right_expr})"
    elif node.value == 'cos':
        # Унарный оператор cos
        return f"cos({tree_to_expression(node.children[0])})"
    elif node.value == 'exp':
        # Унарный оператор exp
        return f"exp({tree_to_expression(node.children[0])})"
    return ""


# Основной алгоритм
def genetic_programming(X, y_true, generations=50):
    population = initialize_population()
    for generation in range(generations):
        population.sort(key=lambda tree: fitness_function(tree, X, y_true))
        next_generation = population[:2]  # сохраняем лучших

        while len(next_generation) < POPULATION_SIZE:
            if random.random() < CROSSOVER_PROBABILITY:
                parent1, parent2 = random.choices(population[:TOURNAMENT_SIZE], k=2)
                child1, child2 = crossover(parent1, parent2)
                next_generation += [child1, child2]
            if random.random() < MUTATION_PROBABILITY:
                next_generation.append(mutation(random.choice(population)))

        population = next_generation

        print(f'Generation {generation}, loss: {tree_to_expression(
            sorted(population, key=lambda tree: fitness_function(tree, X, y_true))[0])}')

    return sorted(population, key=lambda tree: fitness_function(tree, X, y_true))[0]


# Пример использования
X = np.random.uniform(low=-5, high=5, size=(100, 10))
y_true = np.apply_along_axis(target_function, axis=1, arr=X)

X = preprocessing.scale(X)
y_true = preprocessing.scale(y_true)

best_tree = genetic_programming(X, y_true)
print("Лучшее дерево:", best_tree)
