import numpy as np
import random
from deap import base, creator, tools, algorithms


# Функция фитнеса (Rastrigin's function)
def fitness_function(chromosome):
    chromosome = np.array(chromosome)  # Преобразуем список в numpy массив
    return 10 * len(chromosome) + np.sum(np.power(chromosome, 2) - 10 * np.cos(2 * np.pi * chromosome)),


# Настройка генетического алгоритма с помощью DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Для минимизации
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Параметры индивидуума (длина хромосомы и диапазон значений)
CHROMOSOME_LENGTH = 2  # Длина хромосомы (количество параметров)
BOUND_LOW, BOUND_HIGH = -5.12, 5.12  # Границы поиска

# Определим хромосому как список реальных чисел в пределах от BOUND_LOW до BOUND_HIGH
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_HIGH)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, CHROMOSOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Функция оценки (фитнес)
toolbox.register("evaluate", fitness_function)

# Операторы генетического алгоритма
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Оператор скрещивания (Blend Crossover)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # Оператор мутации (гауссовское распределение)
toolbox.register("select", tools.selTournament, tournsize=3)  # Турнирная селекция

# Параметры алгоритма
toolbox.register("map", map)


# Основной процесс ГА
def main():
    random.seed(42)

    # Инициализация популяции
    population = toolbox.population(n=100)

    # Статистика для отслеживания прогресса
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Запуск генетического алгоритма
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                              stats=stats, verbose=True)

    # Найти и вывести лучший индивидуум
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual is {best_individual}, with fitness: {best_individual.fitness.values[0]}")


if __name__ == "__main__":
    main()
