import pandas as pd
import numpy as np
import genalg

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

print(f"Load data and split it into train and test with {100 - int(TEST_SIZE * 100)}/{int(TEST_SIZE * 100)}")

df = pd.read_csv("nasa_data.csv")
L = np.array(df["L"])
E = np.array(df["Ef"])

x_train, x_test, y_train, y_test = train_test_split(L, E, test_size=TEST_SIZE, random_state=1337)

print("Loading completed!\n")


def get_user_input(prompt, default_value, cast_func):
    user_input = input(f"{prompt} (default: {default_value}): ")
    if user_input.strip() == "":
        return default_value
    try:
        return cast_func(user_input)
    except ValueError:
        print("Invalid input. Using default value.")
        return default_value


MAX_GENERATIONS = get_user_input("Enter max generations", 100, int)
POPULATION_SIZE = get_user_input("Enter population size", 200, int)
CROSSOVER_PROBABILITY = get_user_input("Enter crossover probability", 0.8, float)
MUTATION_PROBABILITY = get_user_input("Enter mutation probability", 0.2, float)

print("\nInput completed. Start algorithm...\n")

genetic_optimizer = genalg.RegressionGeneticAlgorithm(POPULATION_SIZE, MAX_GENERATIONS, CROSSOVER_PROBABILITY,
                                                      MUTATION_PROBABILITY)
genetic_optimizer.fit(x_train, y_train)

print("\nFitting completed!")
genetic_optimizer.evaluate_best_solution(x_test, y_test)
