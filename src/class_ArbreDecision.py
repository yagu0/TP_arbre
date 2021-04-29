import graphviz as gv
import numpy as np
import pandas as pd
from src.class_Noeud import Noeud

class ArbreDecision:
    # critere = 'g', 'e' ou 'm'
    def __init__(self, data, classIndex=0, critere='g', dataTypes={}):
        self.data = data
        if len(dataTypes) >= 1:
            # Copie "profonde" pour ne pas modifier le jeu de données en paramètre:
            self.data = data.copy(deep=True)
            for dt in dataTypes.items():
                self.data[data.axes[1][dt[0]]] = self.data[data.axes[1][dt[0]]].astype(dt[1])
        self.classIndex = classIndex
        self.critere = critere
        self.racine = Noeud(self.data, classIndex, range(0, len(data)))

    # Construction de l'arbre de décision
    def learn(self):
        self.racine.learn(self.critere)
        self.racine.adjust_alpha()

    # Retourne toutes les valeurs critiques de alpha
    # TODO: afficher aussi nsplit (comme dans rpart)
    def get_alphas(self):
        alphas = self.racine.get_alphas()
        return list( np.sort(np.unique(alphas)) )

    # Retourne la classe de l'example(s) donné en parametre
    def predict(self, exemple, alpha=0.0):
        if isinstance(exemple, pd.DataFrame):
            return exemple.apply(lambda row: self.racine.predict(row, alpha), axis=1)
        return self.racine.predict(exemple, alpha)

    # Affichage textuel de l'arbre
    def affiche(self, alpha=0.0):
        self.racine.affiche(alpha, 0)

    # Dessine l'arbre
    def plot(self, alpha=0.0, inline=True):
        gtree = gv.Digraph(format='png')
        gobj = self.racine.to_graph(gtree, 'A', alpha)
        if not inline:
            gobj.render(view=True)
        return gobj
