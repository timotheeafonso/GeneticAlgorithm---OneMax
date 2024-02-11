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
P_MUTATION = 1.0
MAX_GENERATIONS = 10000
NB_RUNS = 30
ALPHA = 0.1
PMIN = 0.125
random.seed()

toolbox = base.Toolbox()

# opérateur qui retourne 0
toolbox.register("zeroOrOne", random.randint, 0, 0)

# fonction mono objectif qui maximise la première composante de fitness (c'est un tuple)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# classe Individual construite avec un containeur list
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# initialisation de la population
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Calcul de la fitness/ fonction evaluate
def leadingOnesFitness(individual):
    sum = 0
    zero_occur = False
    for i in individual:
        if zero_occur==False:
            sum += i
        if i == 0:
            zero_occur=True
    return sum,  # return a tuple

toolbox.register("evaluate", leadingOnesFitness)

# Sélection tournoi taille 3
toolbox.register("select", tools.selTournament, tournsize=3)

# Uniform crossover
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# Mutation Bit-Flip
toolbox.register("bitflip", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

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

OPERATORS = [bit_flip,one_flip,trois_flips,cinq_flips]#,"negative-flip"]

def choose_operator(PROB_DIST):
    return random.choices(OPERATORS, weights=PROB_DIST)[0]

# Fonction pour mettre à jour les probabilités
def update_probabilities(UTILITIES,PROB_DIST):
    for opera in OPERATORS:
        if sum(utilities_op[-1] for utilities_op in UTILITIES)!=0:
            PROB_DIST[OPERATORS.index(opera)] = PMIN + (1-len(OPERATORS)*PMIN)*(UTILITIES[OPERATORS.index(opera)][-1])/sum(utilities_op[-1] for utilities_op in UTILITIES)
        else:
            pass
# Fonction pour calculer l'utilité de chaque opérateur
def calculate_utility(operator,UTILITIES,GAIN):
    for op in OPERATORS:
        if op != operator:
            pass
        else:
            ui = (1-ALPHA)*UTILITIES[OPERATORS.index(operator)][-1]+ALPHA*((GAIN[OPERATORS.index(operator)][-1]))
            UTILITIES[OPERATORS.index(operator)].append(ui)

# définition le l'opérateur de mutation par défaut pour l'AG simple
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

# sélectionner le moins bon avec la méthode selWorst
toolbox.register("worst",tools.selWorst, fit_attr='fitness')

def insertion_best_fitness(population, offspring):
    worst = toolbox.worst(population, 1)
    for ind in offspring:
        if (ind.fitness.values[0] > worst[0].fitness.values[0]):
            population.remove(worst[0])
            population.append(ind)
            worst = toolbox.worst(population, 1)
    return population

def main():
    maxFitness_history = []
    meanFitness_history = []
    proba_distrib_history = []

    for i in range(NB_RUNS):
        # accumulateurs pour les statistiques
        PROB_DIST = [0.25 for _ in range(0,len(OPERATORS))]
        GAIN = [[0.] for _ in range(0,len(OPERATORS))]
        UTILITIES = [[0.] for _ in range(0,len(OPERATORS))]
        maxFitnessValues = []
        meanFitnessValues = []
        proba_distrib_values = [[0.25]  for _ in range(0,len(OPERATORS))]
        population = toolbox.populationCreator(n=POPULATION_SIZE)

        # Récupération des valeurs de fitness
        fitnessValues = list(map(toolbox.evaluate, population))
        for individual, fitnessValue in zip(population, fitnessValues):
            individual.fitness.values = fitnessValue
        fitnessValues = [individual.fitness.values[0] for individual in population]
        meanFitness = sum(fitnessValues) / len(population)
        #boucle principale évolution
        generationCounter = 0
        while generationCounter < MAX_GENERATIONS:
            #if generationCounter>0.8*MAX_GENERATIONS:
            #    toolbox.register("evaluate", oneMaxFitnessInv)
            old_fitness = meanFitness
            # sélection des parents mode steady state
            offspring = toolbox.select(population,2)
            offspring = list(map(toolbox.clone, offspring))
            
            operator = choose_operator(PROB_DIST)
            for individual in offspring:
                if random.random() < P_MUTATION:  
                    operator(individual)
                del individual.fitness.values
            
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
            gain = max(0,meanFitness - old_fitness)
            for op in OPERATORS:
                if op != operator:
                    GAIN[OPERATORS.index(op)].append(0.)
                else:
                    GAIN[OPERATORS.index(operator)].append(gain)
            calculate_utility(operator, UTILITIES,GAIN)
            update_probabilities(UTILITIES,PROB_DIST)
            for i in range(len(OPERATORS)):
                proba_distrib_values[i].append(PROB_DIST[i])
            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)
            generationCounter = generationCounter + 1
        # Enregistrement des datas pour les graphiques
        maxFitness_history.append(maxFitnessValues)
        meanFitness_history.append(meanFitnessValues)
        proba_distrib_history.append(proba_distrib_values)
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

    mean_proba_op = [[] for _ in range(len(OPERATORS)) ]
    for op in range(len(OPERATORS)):
        for g in range(MAX_GENERATIONS):
            somme_proba_op=0
            for r in range(NB_RUNS):
                somme_proba_op += proba_distrib_history[r][op][g]
            mean_proba_op[op].append(somme_proba_op/NB_RUNS)

    # Génération d'un graphique
    sns.set_style("whitegrid")
    plt.plot(Mean_maxFitnessValues, color='blue', label='Fitness max')
    plt.plot(Mean_meanFitnessValues, color='orange', label='Fitness moyenne')

    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness moyenne et max en fonction des générations sur' + str(NB_RUNS) + ' runs.')
    plt.show()

    sns.set_style("whitegrid")
    plt.plot(mean_proba_op[0], color='blue', label='bit-flip')
    plt.plot(mean_proba_op[1], color='red', label='1-flip')
    plt.plot(mean_proba_op[2], color='green', label='3-flip')
    plt.plot(mean_proba_op[3], color='black', label='5-flip')
    #plt.plot(mean_proba_op[4], color='orange', label='Operateur inutile')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Distribution')
    plt.title('Distribution des opérateurs de mutation en fonction des générations sur ' + str(NB_RUNS) + ' runs.')
    plt.show()

if __name__ == '__main__':
    main()

