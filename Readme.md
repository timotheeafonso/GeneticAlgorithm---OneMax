# Algorithmes intelligents pour l’aide à la décision

Algorithmes génétiques appliqué au problème du One-Max  implémentés en Python avec le framework DEAP.

- oneMaxSteadyState.py : algorithme génétique steady state classique. On peut comparé les opérateurs de mutation, de croisement, de sélection et la taille de la population (nécéssite de modifier la liste OPERATEURS dans main())
- oneMaxEstimateDistrib.py : algorithme génétique à estimation de distribution
- oneMaxCompactAlgo.py : algorithme génétique compact 
- oneMaxAdaptative.py : algorithme génétique steady state avec roulette adaptative pour la sélection des opérateurs de mutation
- oneMaxUCB.py : algorithme génétique steady state avec UCB pour la sélection des opérateurs de mutation 
- oneMaxMasqueAdaptative.py et leadingOnesAdaptative.py : algorithme génétique steady state avec roulette adaptative pour la sélection des opérateurs de mutation pour les problème du one-max avec masque sur la fonction d'évaluation et leading ones
- UCB_vs_Adaptative.py et Adaptative_vs_NonAdaptative.py : comparaison de la roulette adaptative avec l'UCB et d'autre algorithme génétique non adaptatif.