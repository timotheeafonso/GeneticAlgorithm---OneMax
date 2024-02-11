from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ONE_MAX_LENGTH = 300
POPULATION_SIZE = 2
MAX_GENERATIONS = 7000
NB_RUNS = 3
c=1
random.seed()
toolbox = base.Toolbox()

# opérateur qui retourne 0 ou 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# fonction mono objectif qui maximise la première composante de fitness (c'est un tuple)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# classe Individual construite avec un containeur list
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# initialisation de la population
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Calcul de la fitness/ fonction evaluate
def oneMaxFitness(individual):
    return sum(individual),
    
toolbox.register("evaluate", oneMaxFitness)

def genere_distib_initiale():
    distrib=[]
    for _ in range(ONE_MAX_LENGTH):
        distrib.append(0.5)
    return distrib

def genere_population_distribution(population,distrib):
    for individu in population:
        for position in range(ONE_MAX_LENGTH):
            if random.random() < distrib[position]:
                individu[position] = 1
            else:
                individu[position] = 0
    return population

def maj_estimation_distribution(population,distrib):
    winner = tools.selBest(population,1)[0]
    loser = tools.selWorst(population,1)[0]
    for position in range(ONE_MAX_LENGTH):
        if winner[position] != loser[position]:
            if winner[position] == 1:
                distrib[position]=max(0.01,distrib[position]+1/ONE_MAX_LENGTH * c)
            else:
                distrib[position]=max(0.01,distrib[position]-1/ONE_MAX_LENGTH * c)
    return distrib

def main():
    maxFitness_history = []
    meanFitness_history = []

    for i in range(NB_RUNS):
        maxFitnessValues = []
        meanFitnessValues = []
        population = toolbox.populationCreator(n=POPULATION_SIZE)
        generationCounter = 0
        distrib = genere_distib_initiale()
        while generationCounter < MAX_GENERATIONS:
            generationCounter = generationCounter + 1
            population = genere_population_distribution(population,distrib)
            fitnessValues =list(map(toolbox.evaluate, population))
            for individual, fitnessValue in zip(population, fitnessValues):
                individual.fitness.values = fitnessValue
            fitnessValues = [individual.fitness.values[0] for individual in population]
            distrib = maj_estimation_distribution(population,distrib)

            # MAJ des listes pour les statistiques
            fitnessValues = [ind.fitness.values[0] for ind in population]
            maxFitness = max(fitnessValues)
            meanFitness = sum(fitnessValues) / len(population)
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)
        maxFitness_history.append(maxFitnessValues)
        meanFitness_history.append(meanFitnessValues)
    Mean_maxFitnessValues = []
    Mean_meanFitnessValues = []
    for g in range(MAX_GENERATIONS):
        somme_max = 0
        somme_mean = 0
        for r in range(NB_RUNS):
            somme_max = somme_max + maxFitness_history[r][g]
            somme_mean = somme_mean + meanFitness_history[r][g]
        Mean_maxFitnessValues.append(somme_max/NB_RUNS)
        Mean_meanFitnessValues.append(somme_mean/NB_RUNS)

    # Génération d'un graphique
    sns.set_style("whitegrid")
    plt.plot(Mean_maxFitnessValues, color='blue', label='fitness max')
    plt.plot(Mean_meanFitnessValues, color='orange', label='fitness moyenne')

    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness moyenne et max en fonction des générations sur ' + str(NB_RUNS) + ' runs.')
    plt.show()

if __name__ == '__main__':
    main()

