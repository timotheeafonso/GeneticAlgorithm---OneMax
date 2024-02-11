import matplotlib.pyplot as plt
import seaborn as sns
from oneMaxUCB import *
from oneMaxAdaptative import *

def main():
    Mean_maxFitnessValuesUCB,Mean_meanFitnessValuesUCB,mean_proba_opUCB = UCB()
    Mean_maxFitnessValuesAda,Mean_meanFitnessValuesAda,mean_proba_opAda = rouletteAdaptative()
    
    # Génération d'un graphique
    sns.set_style("whitegrid")
    plt.plot(Mean_meanFitnessValuesUCB, color='blue', label='Fitness moyenne UCB')
    plt.plot(Mean_meanFitnessValuesAda, color='orange', label='Fitness moyenne Roulette adaptative')

    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness moyenne et max en fonction des générations sur' + str(NB_RUNS) + ' runs.')
    plt.show()

if __name__ == '__main__':
    main()

