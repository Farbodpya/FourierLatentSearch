import time
import matplotlib.pyplot as plt
from benchmark_functions import sphere, rastrigin, rastrigin_ii, ackley, griewank, zakharov
from ga_tools import tournament_selection, crossover, mutate
from fourier_projection import to_latent_fourier_space, from_latent_fourier_space
import numpy as np

# Parameters
dim = 5000
latent_dim = 1
pop_size = 50
generations = 200
mutation_rate = 0.2
tournament_k = 3

# GA Fitness Function Wrapper
def fitness_wrapper(individual_latent, func):
    decoded = from_latent_fourier_space(individual_latent, dim)
    return func(decoded)

def run_ga(func):
    population = [to_latent_fourier_space(np.random.randn(dim)) for _ in range(pop_size)]
    best_costs = []

    for _ in range(generations):
        fitness = [fitness_wrapper(ind, func) for ind in population]
        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, fitness, tournament_k)
            p2 = tournament_selection(population, fitness, tournament_k)
            child = crossover(p1, p2)
            if np.random.rand() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
        best_costs.append(np.min(fitness))
    return best_costs

# Run & Plot
functions = {
    'Sphere': sphere,
    'Rastrigin': rastrigin,
    'Rastrigin II': rastrigin_ii,
    'Ackley': ackley,
    'Griewank': griewank,
    'Zakharov': zakharov
}

plt.figure(figsize=(12, 6))

for name, func in functions.items():
    print(f"\nRunning on {name}...")
    start = time.time()
    costs = run_ga(func)
    elapsed = time.time() - start
    print(f"  Final Best Cost: {costs[-1]:.6e}")
    print(f"  Time Taken: {elapsed:.2f} seconds")
    plt.plot(costs, label=name)

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Best Cost (log scale)")
plt.title("Fourier-Latent GA: Cost vs Iteration (7 Benchmark Functions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fourier_latent_ga_7functions.png", dpi=300)
plt.show()
