import numpy as np
import matplotlib.pyplot as plt
import random
import copy


# Partie 1 #

def cree_matrice_transition1():
    """
    Fonction qui représente la matrice de probabilité des transitions de la partie 1 du sujet.

    :param: None

    :return: une matrice correspondant à la matrice de probabilité des transitions
    :rtype: numpy.array
    """
    #   S    I    R
    # S 0.92 0.08 0.00
    # I 0.00 0.93 0.07
    # R 0.00 0.00 1.00
    return np.array([[0.92, 0.08, 0.00],
                     [0.00, 0.93, 0.07],
                     [0.00, 0.00, 1.00]], dtype=float)


def cree_matrice_initiale1():
    """
        Fonction qui représente la matrice initiale de la partie 1 du sujet.

        :param: None

        :return: une matrice correspondant a la matrice initiale des probabilites
        :rtype: numpy.array
    """
    # 0.9 sain , 0.1 infecter , 0 gueri car non malade
    return np.array([0.9, 0.1, 0.0], dtype=float)


def matrice_stochastique(matrice):
    """
        Fonction qui vérifie si une matrice est stochastique.

        :param matrice: matrice que l'on souhaite vérifier s'il elle est stochastique
        :type matrice: numpy.array

        :return: True si la matrice est stochastique et False sinon
        :rtype: bool
    """
    # Une est stochastique si :
    # - pour tous i < M et j < M avec M = longeure matrice , M[i,j] >= 0 
    # - la somme des éléments de chaque ligne vaut 1

    for ligne in matrice:

        # Test tous les éléments d'une ligne sont positif
        if not np.array(map(est_positif, ligne)).all():
            return False
        # print(ligne)
        # Test somme ligne égale 1
        if sum(ligne) != 1.:
            return False

    return True


def est_positif(x):
    """
        Fonction qui vérifie si un nombre est positif.

        :param x: réel que l'on souhaite verifier s'il est positif
        :type x: float

        :return: True si x est positif et False sinon
        :rtype: bool
    """
    return x >= 0.


def modelisation_population(initial, transition, temps, population):
    """
        Fonction qui calcule a chaque instant temps, l'etat de la population.
        Retourne un dictionnaire possédant 3 clefs : "sain" , "infecter" et "gueri".
        Chaque élément de ce dictionnaire possede une liste contenant l'etat de la population a chaque instant.
        Les listes sont rangées chronologiquement.

        :param initial: matrice de probabilite de la matrice initiale
        :param transition: matrice de probabilite de la matrice de transition
        :param temps: nombre d'iterations de l'évolution que l'on souhaite observer
        :param population: nombre de personnes que l'on observe

        :type initial: numpy.array
        :type transition: numpy.array
        :type temps: int
        :type population: int

        :return: dictionnaire possédant 3 clefs : "sain" , "infecter" et "gueri".
        :rtype: dict
    """
    dico = {"sain": [(initial[0] * population)], "infecter": [(initial[1] * population)],
            "gueri": [(initial[2] * population)]}
    temporaire = initial

    for i in range(1, temps):
        temporaire = temporaire @ transition  # @ est le produit matriciel
        dico["sain"].append(temporaire[0] * population)
        dico["infecter"].append(temporaire[1] * population)
        dico["gueri"].append(temporaire[2] * population)

    return dico


def individus_infecter(liste):
    """
        Fonction qui retourne le nombre maximum d'individus infectés lors d'une modelisation.

        :param liste: liste contenant les différentes populations infectées
        :type liste: numpy.array

        :return: nombre maximum d'individus infectés lors d'une modélisation et -1 si la liste est vide
        :rtype: int
    """
    if len(liste) >= 1:
        maximum = liste[0]
        index = 0
        for i in range(1, len(liste)):
            if liste[i] > maximum:
                maximum = liste[i]
                index = i

        return maximum
    return -1


def pic_epidemie(liste):
    """
        Fonction qui retourne le temps ou le nombre maximum d'individus qui a été infectés lors d'une modelisation.

        :param liste: liste contenant les différentes populations infectés
        :type liste: numpy.array

        :return: le temps ou le nombre maximum d'individus qui a été infectés lors d'une modelisation
                et -1 si la liste est vide
        :rtype: int
    """
    if len(liste) >= 1:
        maximum = liste[0]
        index = 0
        for i in range(1, len(liste)):
            if liste[i] > maximum:
                maximum = liste[i]
                index = i

        return index
    return -1

def longeur_infection(initial, transition, temps):
    moyenne = 0

    #On réalise temps fois l'experience afin d'avoir une meilleur estimation
    for i in range(temps):

        temporaire = np.dot(initial , transition )
        aleatoire = random.uniform(0,1) #avoir un tirage uniforme
        longeure = 0
    
        #temporaire[1] est la probabilite du nombre d'infecter
        while temporaire[2] < aleatoire and longeure < temps :
            longeure +=1
            temporaire = np.dot(temporaire , transition ) # @ est le produit matriciel
            aleatoire = random.uniform(0,1)
            
        
        #print(longeure)
        moyenne += longeure
    
    return moyenne / temps


def proportion_individus_t_grand(dico):
    """
        Fonction qui nous informes de la proportion des individus lorsque le temps est grand.
        En fonction d'un dictionnaire possédant 3 clefs : "sain" , "infecter" et "gueri".
        Chaque élément de ce dictionnaire possede une liste contenant l'etat de la population a chaque instant.
        Les listes sont rangées chronologiquement.
        Retourne une liste de taille 3 telle que :
        - liste[0] = proportions individus sain quand t grand ;
        - liste[1] = proportions individus infecter quand t grand ;
        - liste[2] = proportions individus gueri quand t grand ;

        :param dico: dictionnaire possédant 3 clefs : "sain" , "infecter" et "gueri".
        :type dico: dict

        :return: liste de taille 3
        :rtype: list
    """
    # Puisque les listes sont rangées par ordre chronologiques pour savoir quand t est grand
    # il suffit de prendre le dernier élément
    sain = dico["sain"][len(dico["sain"]) - 1]
    infecter = dico["infecter"][len(dico["infecter"]) - 1]
    gueri = dico["gueri"][len(dico["gueri"]) - 1]
    return [sain, infecter, gueri]


def dessiner_graphe_modelisation(dico, temps, population):
    """
        Fonction qui dessine un graphe en fonction de la modelisation faite.

        :param dico: dictionnaire contenant les états des populations a chaque instant lors de la modélisation
        :param temps: nombre d'iteration de l'évolution que l'on souhaite observer
        :param population: nombre de personne que l'on observe

        :type dico: dict
        :type temps: int
        :type population: int

        :return: None
    """

    liste_temps = [i for i in range(temps)]
    # liste_population = [i for i in range(0,population,population/8)]
    # Création figure
    plt.figure()
    dict_sain = [dico["sain"][i] for i in range(temps)]
    dict_infecter = [dico["infecter"][i] for i in range(temps)]
    dict_gueri = [dico["gueri"][i] for i in range(temps)]

    plt.plot(liste_temps, dict_sain, label='sain')
    plt.plot(liste_temps, dict_infecter, label='infecté')
    plt.plot(liste_temps, dict_gueri, label='guéri')
    plt.title("Nombre d’individus dans chaque état en fonction du temps")
    plt.xlabel("Temps")
    plt.ylabel("Nombre de personnes dans chaque catégorie")
    plt.legend()
    plt.show()


def dessine_evolution_temps(dico, temps, population):
    """
        Fonction qui dessine un graphe en fonction de la modelisation faite à 4 temps différents.

        :param dico: dictionnaire contenant les états des populations a chaque instant lors de la modélisation
        :param temps: nombre d'iteration de l'évolution que l'on souhaite observer
        :param population: nombre de personne que l'on observe

        :type dico: dict
        :type temps: int
        :type population: int

        :return: None
    """
    dessiner_graphe_modelisation(dico, temps // 10, population)
    dessiner_graphe_modelisation(dico, temps // 5, population)
    dessiner_graphe_modelisation(dico, temps // 2, population)
    dessiner_graphe_modelisation(dico, temps, population)


# Partie 2 #

def cree_matrice_transition2():
    """
        Fonction qui représente la matrice de probabilité de transition du modèle ergodique (partie 2).

        :param: None

        :return: une matrice correspondant a la matrice de probabilité des transitions
        :rtype: numpy.array
    """
    #   S    I    R
    # S 0.92 0.08 0.00
    # I 0.00 0.93 0.07
    # R 0.04 0.00 0.96
    return np.array([[0.92, 0.08, 0.00],
                     [0.00, 0.93, 0.07],
                     [0.04, 0.00, 0.96]], dtype=float)


def cree_matrice_initiale2():
    """
        Fonction qui représente la matrice initiale du modèle ergodique (partie 2).

        :param: None

        :return: une matrice correspondant a la matrice initiale des probabilites
        :rtype: numpy.array
    """
    # 1 sain , 0 infecter , 0 gueri car non malade
    return np.array([1.0, 0.0, 0.0], dtype=float)


def cree_matrice_initiale_choix(sain, infecter, gueri):
    """
        Fonction qui représente la matrice initiale du modèle ergodique (partie 2) selon une répartition donnée.

        :param sain: poucentage de personnes saines dans la population
        :param infecter: poucentage de personnes infectées dans la population
        :param gueri: poucentage de personnes guéries dans la population

        :type sain: float
        :type infecter: float
        :type gueri: float

        :return: une matrice correspondant a la matrice initiale des probabilites
        :rtype: numpy.array
    """
    return np.array([sain, infecter, gueri], dtype=float)


def resolution_equation(matrice_transition, matrice_initiale, population):
    """
        Fonction qui retourne la distribution stationnaire en fonction de la matrice initiale et de la population.
        On fait une approximation de la valeur.

        :param matrice_transition: matrice de probabilite de la matrice de transition
        :param matrice_initiale: matrice de probabilite de la matrice de initiale
        :param population: nombre de personnes que l'on observe

        :type matrice_transition: numpy.array
        :type matrice_initiale: numpy.array
        :type population: int

        :return: matrice stationnaire correspondant a la distribution de la population
        :rtype: numpy.array
    """
    matrice = copy.deepcopy(matrice_transition)
    temp = matrice

    for i in range(150):
        temp = matrice @ temp

    return (matrice_initiale @ temp) * population


# Partie 3 #

def cree_matrice_transition3():
    """
        Fonction qui représente la matrice de probabilité de transition de la partie 3 du sujet.

        :param: None

        :return: une matrice correspondant a la matrice de probabiliter des transitions
        :rtype: numpy.array
    """
    #   S    I    R
    # S 0.98 0.02 0.00
    # I 0.00 0.93 0.07
    # R 0.04 0.00 0.96
    return np.array([[0.98, 0.02, 0.00],
                     [0.00, 0.93, 0.07],
                     [0.04, 0.00, 0.96]], dtype=float)


def cree_matrice_initiale3():
    """
        Fonction qui représente la matrice initiale de la partie 3 du sujet.

        :param: None

        :return: une matrice correspondant a la matrice initiale des probabilites
        :rtype: numpy.array
    """
    # 1 sain , 0 infecter , 0 gueri car non malade
    return np.array([1.0, 0.0, 0.0], dtype=float)


def modelisation_population_alterner(initial, transition1, transition2, proba1, proba2, temps, population):
    """
    Fonction qui calcule a chaque instant temps, l'etat de la population.
    Dans cette fonction nous allons alterner les matrices de transition1 et transition2
    en fonction des proba1 et proba2.

    :param initial: matrice de probabilite de la matrice initiale
    :param transition1: matrice de probabilite de la matrice de transition1 par laquelle on va commencer notre programme
    :param transition2: matrice de probabilite de la matrice de transition2 par laquelle on va alterner avec transition1
    :param temps: nombre d'iterations de l'evolution que l'on souhaite observer
    :param population: nombre de personnes que l'on observe
    :param proba1: probabilité d'alterner et de passer de transition1 vers transition2
    :param proba2: probabilité d'alterner et de passer de transition2 vers transition1

    :type initial: numpy.array
    :type transition1: numpy.array
    :type transition2: numpy.array
    :type temps: int
    :type population: int
    :type proba1: float
    :type proba2: float

    :return: un dictionnaire possédant 3 clefs : "sain" , "infecter" et "gueri".
    :rtype: dict
    """
    dico = {"sain": [(initial[0] * population)], "infecter": [(initial[1] * population)],
            "gueri": [(initial[2] * population)]}
    temporaire = initial
    transition = transition1
    for i in range(1, temps):

        if temporaire[1] >= proba1:
            transition = transition2

        if temporaire[0] <= proba2:
            transition = transition1

        temporaire = temporaire @ transition  # @ est le produit matriciel
        dico["sain"].append(temporaire[0] * population)
        dico["infecter"].append(temporaire[1] * population)
        dico["gueri"].append(temporaire[2] * population)

    return dico


# Fonctions d'affichage #

def affichage_t_grand(dico):
    """
        Fonction d'affichage du nombre d'individus sains, infectés et guéris d'une population pour t grand.

        :param dico: dictionnaire contenant les états des populations a chaque instant lors de la modélisation
        :type dico: dict

        :return: None
    """
    liste = proportion_individus_t_grand(dico)
    print("Lorsque t est grand : ")
    print("\t - nombre d'individus sains : {} ".format(round(liste[0])))
    print("\t - nombre d'individus infectés : {} ".format(round(liste[1])))
    print("\t - nombre d'individus guéris : {} \n".format(round(liste[2])))


def affichage_pic_epidemie(dico):
    """
        Fonction d'affichage du pic de l'épidémie qui arrive au temps t avec son nombre d'infectés.

        :param dico: dictionnaire contenant les états des populations a chaque instant lors de la modélisation
        :type dico: dict

        :return: None
    """
    pic = pic_epidemie(dico["infecter"])
    individus = individus_infecter(dico["infecter"])
    print("Lorsque le pic de l'épidémie arrive au temps t = {}, "
          "il y a {} infectés dans la population. \n".format(pic, int(individus)))


def affichage_stationnaire_convergence(transition, initiale, population):
    """
        Fonction d'affichage de la matrice stationnaire et de la convergence de la population.

        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale
        :param population: nombre de personnes dans la population observée

        :type transition: numpy.array
        :type initiale: numpy.array
        :type population: int

        :return: None
    """
    affichage_stationnaire(transition, initiale, population)
    affichage_convergence(transition, initiale, population)


def affichage_stationnaire(transition, initiale, population):
    """
        Fonction d'affichage de la matrice stationnaire.

        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale
        :param population: nombre de personnes dans la population observée

        :type transition: numpy.array
        :type initiale: numpy.array
        :type population: int

        :return: matrice stationnaire
    """
    stationnaire = resolution_equation(transition, initiale, population)
    print("Matrice stationnaire : \n{} \n".format(stationnaire))

    return stationnaire


def affichage_convergence(transition, initiale, population):
    """
        Fonction d'affichage de la convergence de la population.

        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale
        :param population: nombre de personnes dans la population observée

        :type transition: numpy.array
        :type initiale: numpy.array
        :type population: int

        :return: None
    """
    stationnaire = resolution_equation(transition, initiale, population)

    print("Convergence de la population : ")
    print("\t - nombre d'individus sains : {} ".format(round(stationnaire[0])))
    print("\t - nombre d'individus infectés : {} ".format(round(stationnaire[1])))
    print("\t - nombre d'individus guéris : {} \n".format(round(stationnaire[2])))


# Partie 4 #

def modele_etude_transition(temps, population, transition, initiale):
    """
        Fonction d'analyse et d'affichage d'un modèle d'étude qui modifie uniquement
        une probabilité de transition dans la matrice de transition.

        :param temps: nombre d'iterations de l'évolution que l'on souhaite observer
        :param population: nombre de personnes dans la population observée
        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale

        :type temps: int
        :type population: int
        :type transition: numpy.array
        :type initiale: numpy.array

        :return: None
    """
    print("Matrice de transition : \n{} \n".format(transition))

    print("Modelisation de la population : ")
    dictionnaire = modelisation_population(initiale, transition, temps, population)

    dessiner_graphe_modelisation(dictionnaire, temps, population)
    affichage_pic_epidemie(dictionnaire)
    affichage_convergence(transition, initiale, population)


def modele_etude_initiale(temps, population, transition, initiale):
    """
        Fonction d'analyse et d'affichage d'un modèle d'étude qui modifie uniquement la matrice initiale.

        :param temps: nombre d'iterations de l'évolution que l'on souhaite observer
        :param population: nombre de personnes dans la population observée
        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale

        :type temps: int
        :type population: int
        :type transition: numpy.array
        :type initiale: numpy.array

        :return: None
    """
    print("Matrice initiale : \n{} \n".format(initiale))

    print("Modelisation de la population : ")
    dictionnaire = modelisation_population(initiale, transition, temps, population)

    dessiner_graphe_modelisation(dictionnaire, temps, population)
    affichage_pic_epidemie(dictionnaire)
    affichage_convergence(transition, initiale, population)


def modele_etude_population(temps, population, transition, initiale):
    """
        Fonction d'analyse et d'affichage d'un modèle d'étude qui modifie uniquement la taille de la population.

        :param temps: nombre d'iterations de l'évolution que l'on souhaite observer
        :param population: nombre de personnes dans la population observée
        :param transition: matrice de probabilite de la matrice de transition
        :param initiale: matrice de probabilite de la matrice initiale

        :type temps: int
        :type population: int
        :type transition: numpy.array
        :type initiale: numpy.array

        :return: None
    """
    print("Taille de la population : {} \n".format(population))

    print("Modelisation de la population : ")
    dictionnaire = modelisation_population(initiale, transition, temps, population)

    dessiner_graphe_modelisation(dictionnaire, temps, population)

    pic = pic_epidemie(dictionnaire["infecter"])
    individus = round(individus_infecter(dictionnaire["infecter"]))
    pourcentage = round(individus / population * 100)

    print("Lorsque le pic de l'épidémie arrive au temps t = {}, "
          "il y a {} ({}%) infectés dans la population. \n".format(pic, individus, pourcentage))

    stationnaire = resolution_equation(transition, initiale, population)

    sains = round(stationnaire[0])
    infectes = round(stationnaire[1])
    gueris = round(stationnaire[2])

    pourcentage_sains = round(sains / population * 100)
    pourcentage_infectes = round(infectes / population * 100)
    pourcentage_gueris = round(gueris / population * 100)

    print("Convergence de la population :")
    print("\t - nombre d'individus sains : {} ({}%) ".format(sains, pourcentage_sains))
    print("\t - nombre d'individus infectés : {} ({}%) ".format(infectes, pourcentage_infectes))
    print("\t - nombre d'individus guéris : {} ({}%) \n".format(gueris, pourcentage_gueris))

