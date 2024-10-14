import numpy as np
import random

def iterarFLO(maxIter, iter, dim, population, fitness, bestSolution, fo, typeProblem, lowerBound, upperBound):
    # Fase 1: Exploración (Hunting Strategy)
    for i in range(population.__len__()):
        CPi = [bestSolution[j] for j in range(population.__len__()) if fitness[j] < fitness[i] and j != i] if typeProblem == 'MIN' else \
              [bestSolution[j] for j in range(population.__len__()) if fitness[j] > fitness[i] and j != i]
        
        if CPi:
            prey = CPi[np.random.randint(len(CPi))]
        else:
            prey = population[np.random.randint(population.__len__())]
        
        I = np.random.choice([1, 2])
        r = np.random.rand(dim)
        
        # Nueva posición basada en la presa (fase de exploración)
        newPositionFL = population[i] + r * (prey - I * population[i])
        newPositionFL = np.clip(newPositionFL, lowerBound, upperBound)

        # Evaluar la nueva posición
        repaired_position, newFitnessFL = fo(newPositionFL)

        # Actualizar si la nueva posición es mejor
        if (typeProblem == 'MIN' and fitness[i] > newFitnessFL) or (typeProblem == 'MAX' and fitness[i] < newFitnessFL):
            population[i] = newPositionFL
            fitness[i] = newFitnessFL

    # Fase 2: Explotación (Moving up the tree)
    for i in range(population.__len__()):
        r2 = np.random.rand(dim)
        
        # Nueva posición para la fase de explotación (subida al árbol)
        newPositionTree = population[i] + (1 - r2) * ((upperBound - lowerBound) / (iter + 1))
        newPositionTree = np.clip(newPositionTree, lowerBound, upperBound)

        # Evaluar la nueva posición en la fase de explotación
        repaired_position, newFitnessTree = fo(newPositionTree)

        # Actualizar si la nueva posición es mejor
        if (typeProblem == 'MIN' and fitness[i] > newFitnessTree) or (typeProblem == 'MAX' and fitness[i] < newFitnessTree):
            population[i] = newPositionTree
            fitness[i] = newFitnessTree

    return np.array(population)
