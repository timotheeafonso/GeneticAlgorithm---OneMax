from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import *

ONE_MAX_LENGTH = 300
POPULATION_SIZE = 20
P_MUTATION = 1.0
MAX_GENERATIONS = 3000
NB_RUNS = 30
ALPHA = 0.1
PMIN = 0.125
random.seed()

toolbox = base.Toolbox()

# opérateur qui retourne 0:
toolbox.register("zero", random.randint, 0, 0)

# fonction mono objectif qui maximise la première composante de fitness (c'est un tuple)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# classe Individual construite avec un containeur list
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zero, ONE_MAX_LENGTH)

# initialisation de la population
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Calcul de la fitness/ fonction evaluate
def oneMaxFitness(individual):
    return sum(individual),  # return a tuple

toolbox.register("evaluate", oneMaxFitness)

# Sélection tournoi taille 3
toolbox.register("select", tools.selTournament, tournsize=3)

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

OPERATORS = [bit_flip,one_flip,trois_flips,cinq_flips]

def choose_operator(PROB_DIST):
    return random.choices(OPERATORS, weights=PROB_DIST)[0]

def normalize_array(GAIN):
    flattened_array = [val for sublist in GAIN for val in sublist]
    min_val = min(flattened_array)
    max_val = max(flattened_array)
    normalized_gain = []

    for array in GAIN:
        if max_val != min_val:
            normalized_array = [(val - min_val) / (max_val - min_val) * (1 - 0.01) + 0.01 for val in array]
        else:
            normalized_array = [0.01] * len(array)
        normalized_gain.append(normalized_array)
    return normalized_gain
  
# Fonction pour mettre à jour les probabilités
def update_probabilities(PROB_DIST,count_use_op,GAIN,generation):
    gain_norm = normalize_array(GAIN)
    mean_gain_all_op = 0
    for gain_opera in gain_norm:
        mean_gain_one_op=0
        for val in gain_opera:
            mean_gain_one_op += val
        mean_gain_one_op/=10
        mean_gain_all_op+=mean_gain_one_op

    index = 0
    for gain_opera in gain_norm:       
        gain_op = 0
        for val in gain_opera:
            gain_op += val
        gain_op/=10
        sum_use_op_choose = max(1, sum(count_use_op[index]))
        PROB_DIST[index]=(gain_op + 0.01 * sqrt(2*log(generation)/sum_use_op_choose)) /mean_gain_all_op
        index +=1

# insertion best fitness
toolbox.register("worst",tools.selWorst, fit_attr='fitness')

def insertion_best_fitness(population, offspring):
    worst = toolbox.worst(population, 1)
    for ind in offspring:
        if (ind.fitness.values[0] > worst[0].fitness.values[0]):
            population.remove(worst[0])
            population.append(ind)
            worst = toolbox.worst(population, 1)
    return population

def UCB():
    # initialisations des acccummulateurs statistiques
    maxFitness_history = []
    meanFitness_history = []
    proba_distrib_history = []
    count_use_op_history = []
    for r in range(NB_RUNS):
        PROB_DIST = [0.25 for _ in range(0,len(OPERATORS))]
        GAIN = [[0.25 for i in range(10)] for _ in range(0,len(OPERATORS))]
        count_use_op = [ [] for _ in range(0,len(OPERATORS))]
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
            generationCounter = generationCounter + 1
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

            population = insertion_best_fitness(population, offspring)
            fitnessValues = [ind.fitness.values[0] for ind in population]
            maxFitness = max(fitnessValues)
            meanFitness = sum(fitnessValues) / len(population)
            gain = max(0,meanFitness - old_fitness)
            GAIN[OPERATORS.index(operator)].pop(0)
            GAIN[OPERATORS.index(operator)].append(gain)

            for i in range(len(OPERATORS)):
                proba_distrib_values[i].append(PROB_DIST[i])
            for op in OPERATORS:
                if op==operator:
                    count_use_op[OPERATORS.index(op)].append(1)
                else:
                    count_use_op[OPERATORS.index(op)].append(0)

            update_probabilities(PROB_DIST,count_use_op,GAIN,generationCounter)

            maxFitnessValues.append(maxFitness)
            meanFitnessValues.append(meanFitness)
        # Enregistrement des datas pour les graphiques
        maxFitness_history.append(maxFitnessValues)
        meanFitness_history.append(meanFitnessValues)
        proba_distrib_history.append(proba_distrib_values)
        count_use_op_history.append(count_use_op)
        
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
    mean_count_op = [[] for _ in range(len(OPERATORS)) ]
    for op in range(len(OPERATORS)):
        for g in range(MAX_GENERATIONS):
            somme_proba_op=0
            somme_count_op=0
            for r in range(NB_RUNS):
                somme_proba_op += proba_distrib_history[r][op][g]
                somme_count_op += count_use_op_history[r][op][g]
            mean_proba_op[op].append(somme_proba_op/NB_RUNS)
            mean_count_op[op].append(somme_count_op/NB_RUNS)
    return Mean_maxFitnessValues,Mean_meanFitnessValues,mean_proba_op

def main():
    Mean_maxFitnessValues,Mean_meanFitnessValues,mean_proba_op = UCB()
   
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
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Distribution')
    plt.title('Distribution des opérateurs sur' + str(NB_RUNS) + ' runs.')
    plt.show()

if __name__ == '__main__':
    main()