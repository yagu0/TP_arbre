import numpy as np
import random
from src.class_ArbreDecision import ArbreDecision

# Division d'un jeu de données en deux: train et test.
# test_size donne la taille de l'ensemble de test (absolue ou pourcentage)
def train_test_split(data, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(data))
    # Deux approches possibles ici: permuter aléatoirement les indices puis prendre les
    # k premiers, ou bien (comme ci-dessous) choisir directement k indices au hasard.
    test_indices = random.sample(data.index.tolist(), test_size)
    return data.drop(test_indices), data.loc[test_indices]

# Retourne le taux de bien classés + taux d'erreur
def eval_predict(arbre, dataTest, alpha=0.0):
    predictions = arbre.predict(dataTest, alpha)
    classif_rate = np.mean(predictions == dataTest[ dataTest.axes[1][arbre.classIndex] ])
    return classif_rate, 1.0 - classif_rate

# Évalue l'erreur sur l'ensemble de test pour chaque valeur critique de alpha
def best_alpha(data, test_size, classIndex, critere):
    trainData, testData = train_test_split(data, test_size)
    a = ArbreDecision(trainData, classIndex, critere)
    a.learn()
    alphaList = a.get_alphas() + [1]
    errors = [ eval_predict(a, testData, alpha)[1] for alpha in alphaList ]
    print("Errors:")
    print(errors)
    print("Alphas:")
    print(alphaList)
    return alphaList[len(alphaList) - 1 - np.argmin(np.array(errors)[::-1])]
