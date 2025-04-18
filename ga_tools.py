import numpy as np

def tournament_selection(pop, fitness, k):
    idx = np.random.choice(len(pop), k)
    return pop[idx[np.argmin([fitness[i] for i in idx])]]

def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

def mutate(ind):
    return ind + np.random.randn(*ind.shape) * 0.1
