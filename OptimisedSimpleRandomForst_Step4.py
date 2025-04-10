import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

data = pd.read_csv('cleaned_data.csv')

features = data[['Open', 'High', 'Low', 'Volume']]
target = data['Close']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    return [
        np.random.randint(50, 200),   # n_estimators
        np.random.randint(5, 50),     # max_depth
        np.random.uniform(0.0001, 1.0),  # min_samples_split
        np.random.randint(1, 20)      # min_samples_leaf
    ]

def repair_individual(individual):
    individual[0] = int(np.clip(round(individual[0]), 50, 200))         # n_estimators
    individual[1] = int(np.clip(round(individual[1]), 5, 50))           # max_depth
    individual[2] = float(np.clip(individual[2], 0.0001, 1.0))           # min_samples_split (float)
    individual[3] = int(np.clip(round(individual[3]), 1, 20))            # min_samples_leaf
    return individual

def evaluate(individual):
    individual = repair_individual(individual)

    n_estimators = individual[0]
    max_depth = individual[1]
    min_samples_split = individual[2]
    min_samples_leaf = individual[3]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    return (mae,)

def custom_mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
    return repair_individual(individual),

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=300)
generations = 10
cx_prob = 0.7
mut_prob = 0.2

algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
                    stats=None, halloffame=None, verbose=True)

best_individual = tools.selBest(population, 1)[0]
best_individual = repair_individual(best_individual)

print(f"\nНай-добър индивидуален модел: {best_individual}")

best_model = RandomForestRegressor(
    n_estimators=best_individual[0],
    max_depth=best_individual[1],
    min_samples_split=best_individual[2],
    min_samples_leaf=best_individual[3],
    random_state=42
)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"Средна абсолютна грешка (MAE): {mae_best:.4f}")
print(f"Средна квадратична грешка (RMSE): {rmse_best:.4f}")
