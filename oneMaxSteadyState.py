# Import des librairies/modules
from deap import base
from deap import creator

from deap import tools
from deap import algorithms

import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ONE_MAX_LENGTH = 300
POPULATION_SIZE = 20
P_CROSSOVER = 1.0
P_MUTATION = 1.0
MAX_GENERATIONS = 1700
NB_RUNS = 30
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
    return sum(individual),  # return a tuple

toolbox.register("evaluate", oneMaxFitness)

# Sélection tournoi taille 3
toolbox.register("select", tools.selTournament, tournsize=3)

# Uniform crossover
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# Mutation Bit-Flip
toolbox.register("bitflip", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

# mutations 1FLip... n flips
def flip(b):
    if (b == 1):
        return 0
    else:
        return 1

def one_flip(individual):
    pos = random.randint(0, ONE_MAX_LENGTH - 1)
    individual[pos] = flip(individual[pos])


def n_flips(individual, n):
    lpos = []
    while (len(lpos) < n):
        pos = random.randint(0, ONE_MAX_LENGTH - 1)
        if (lpos.count(pos) == 0):
            lpos.append(pos)
    for pos in lpos:
        individual[pos] = flip(individual[pos])

def trois_flips(individual):
    n_flips(individual,3)

def cinq_flips(individual):
    n_flips(individual,5)

def bit_flip(individual):
    for i in range(len(individual)):
        if random.random() < 1/ONE_MAX_LENGTH:                        
            individual[i] = flip(individual[i])

def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def cxUniform(ind1, ind2, indpb=0.5):
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

# définition le l'opérateur de mutation par défaut pour l'AG simple
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

# sélectionner le moins bon avec la méthode selWorst
toolbox.register("worst",tools.selWorst, fit_attr='fitness')
# insertion best fitness
def insertion_best_fitness(population, offspring):
    worst = toolbox.worst(population, 1)
    for ind in offspring:
        if (ind.fitness.values[0] > worst[0].fitness.values[0]):
            population.remove(worst[0])
            population.append(ind)
            worst = toolbox.worst(population, 1)
    return population

def steadyState(operateurs):
    for o in operateurs:
        POPULATION_SIZE = 30
        selection=tools.selTournament
        croisement = cxUniform
        fcn_muataion = bit_flip
        if "selection" in o:
            selection = o["selection"]
        elif "population" in o:
            POPULATION_SIZE = o["population"]
        elif "croisement" in o:
            croisement = o["croisement"]
        elif "mutation" in o:
            fcn_muataion = o["mutation"]

        maxFitness_history = []
        meanFitness_history = []
        for i in range(NB_RUNS):
            # accumulateurs pour les statistiques
            maxFitnessValues = []
            meanFitnessValues = []
            population = toolbox.populationCreator(n=POPULATION_SIZE)

            # Récupération des valeurs de fitness
            fitnessValues = list(map(toolbox.evaluate, population))
            for individual, fitnessValue in zip(population, fitnessValues):
                individual.fitness.values = fitnessValue
            fitnessValues = [individual.fitness.values[0] for individual in population]

            #boucle principale évolution
            generationCounter = 0
            while generationCounter < MAX_GENERATIONS:
                generationCounter = generationCounter + 1
                # sélection des parents mode steady state
                if selection == tools.selTournament:
                    offspring = selection(population,2,3)
                else :
                    offspring = selection(population,2)
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < P_CROSSOVER:
                        croisement(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                # mutation
                for mutant in offspring:
                    if random.random() < P_MUTATION:
                        fcn_muataion(mutant)
                        del mutant.fitness.values
                # Récupération des individus dont la fitness a changé (non encore calculée)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                # calcul des valeurs de fitness
                fitnesses = map(toolbox.evaluate, invalid_ind)
                # MAJ des valeurs de fitness
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                # insertion dans la population
                population = insertion_best_fitness(population, offspring)

                # MAJ des listes pour les statistiques
                fitnessValues = [ind.fitness.values[0] for ind in population]
                maxFitness = max(fitnessValues)
                meanFitness = sum(fitnessValues) / len(population)
                maxFitnessValues.append(maxFitness)
                meanFitnessValues.append(meanFitness)
            # Enregistrement des datas pour les graphiques
            maxFitness_history.append(maxFitnessValues)
            meanFitness_history.append(meanFitnessValues)

        # préparation des datas pour l'affichage
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
        o["values"] = Mean_maxFitnessValues

def main():
    '''
    OPERATEURS=[
        {"selection" : tools.selTournament, "nom" : "Tournois", "color" : "blue", "values" : []},
        {"selection" : tools.selBest, "nom" : "Meilleur", "color" : "red", "values" : []},
        {"selection" : tools.selRoulette, "nom" : "Roulette", "color" : "green", "values" : []},
        {"selection" : tools.selRandom, "nom" : "Aléatoire", "color" : "black", "values" : []},
    ]
    OPERATEURS=[
        {"croisement" : cxTwoPoint, "nom" : "Two point", "color" : "blue", "values" : []},
        {"croisement" : cxOnePoint, "nom" : "One point", "color" : "red", "values" : []},
        {"croisement" : cxUniform, "nom" : "Uniform", "color" : "green", "values" : []}
    ]
    
    '''
    OPERATEURS=[
        {"mutation" : bit_flip, "nom" : "bit_flip", "color" : "blue", "values" : []},
        {"mutation" : one_flip, "nom" : "one_flip", "color" : "red", "values" : []},
        {"mutation" : trois_flips, "nom" : "trois_flip", "color" : "green", "values" : []},
        {"mutation" : cinq_flips, "nom" : "cinq_flip", "color" : "black", "values" : []},
    ]
    '''
    OPERATEURS=[
        {"population" : 20, "nom" : "20 individus", "color" : "blue", "values" : []},
        {"population" : 30, "nom" : "30 individus", "color" : "red", "values" : []},
        {"population" : 40, "nom" : "40 individus", "color" : "green", "values" : []},
        {"population" : 50, "nom" : "50 individus", "color" : "black", "values" : []},
    ]
    '''
    
    steadyState(OPERATEURS)

    sns.set_style("whitegrid")
    for o in OPERATEURS:
        plt.plot(o["values"], color = o["color"], label=o["nom"])
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness moyenne sur ' + str(NB_RUNS) + ' runs.')
    plt.show()

if __name__ == '__main__':
    main()

