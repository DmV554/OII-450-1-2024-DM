import numpy as np
import random

def iterarFLO(maxIter, iter, dimension, population, fitness, bestSolution, lowerBound=0, upperBound=1):
    for i in range(population.__len__()):
        CPi = [bestSolution[j] for j in range(population.__len__()) if fitness[j] < fitness[i] and j != i]
        
        if CPi:
            prey = CPi[np.random.randint(len(CPi))]
        else:
            prey = population[np.random.randint(population.__len__())]
        
        I = np.random.choice([1, 2])
        r = np.random.rand(dimension)
        newPositionFL = population[i] + r * (prey - I * population[i])
        newPositionFL = np.clip(newPositionFL, lowerBound, upperBound)

        if fitness[i] > np.sum(newPositionFL):
            population[i] = newPositionFL

        r2 = np.random.rand(dimension) * 2
        newPositionTree = population[i] + (1 - r2) * ((upperBound - lowerBound) / (iter+1))
        newPositionTree = np.clip(newPositionTree, lowerBound, upperBound)

        if fitness[i] > np.sum(newPositionTree): 
            population[i] = newPositionTree

    return np.array(population)