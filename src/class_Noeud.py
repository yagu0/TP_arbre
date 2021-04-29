from src.code_utils import est_manquant, smart_print
import itertools
import math
import numpy as np
import pandas as pd
import random

class Noeud:
    def __init__(self, data, classIndex, indices,
                 attribut=None, seuil=None, gauche=None, droit=None, alpha=None):
        # Données (passage par référence)
        self.data = data
        # Numéro de la colonne donnant le label
        self.classIndex = classIndex
        # Indices des éléments présents à ce noeud
        self.indices = indices
        # Classe (majoritaire): np.array de classes ex-aequos
        self.classe, self.error = self.classe_majoritaire()
        # Champs suivants seulement pour un noeud interne (après split):
        self.attribut = attribut  #numéro de l'attribut
        self.seuil = seuil        #point de comparaison du split (>= / <= ou != ==)
        self.alpha = alpha        #valeur critique du paramètre d'élaguage
        self.gauche = gauche      #sous-arbre gauche (valeurs <= au seuil)
        self.droit = droit        #sous-arbre droit (valeurs > au seuil)

    # Renvoie la classe majoritaire dans un dataset (np.array, potentiels ex-aequos)
    def classe_majoritaire(self):
        classes, counts = np.unique(self.data.iloc[self.indices,self.classIndex], return_counts=True)
        maxOccurrences = np.max(counts)
        # Note: l'erreur est calculée en absolue (division par n)
        return classes[counts == maxOccurrences], (len(self.indices) - maxOccurrences) / len(self.data)

    def est_feuille(self):
        return self.attribut == None #pas de split => feuille

    def get_classe(self):
        # En cas d'ex-aequos, on retourne une classe au hasard:
        return random.choice(self.classe)

    def est_homogene(self):
        # Un noeud est homogène s'il ne contient qu'une seule classe, ou,
        # de manière équivalente, si l'erreur réalisée est nulle.
        return self.error == 0.0

    # Critères d'information (respectivement 'g', 'e' et 'm')
    def compute_gini(self, probas):
        return 1.0 - sum(probas * probas)
    def compute_entropy(self, probas):
        return - sum(probas * np.log(probas))
    def compute_misclass(self, probas):
        return 1.0 - np.max(probas)

    # Retourne le gain d'information d'une division en indLeft / indRight
    def eval_critere(self, critere, indLeft, indRight):
        _, leftCounts = np.unique(self.data.iloc[indLeft,self.classIndex], return_counts=True)
        _, rightCounts = np.unique(self.data.iloc[indRight,self.classIndex], return_counts=True)
        effectifs = [sum(leftCounts), sum(rightCounts)]
        leftProbas = leftCounts / effectifs[0]
        rightProbas = rightCounts / effectifs[1]
        critVal = []
        if critere == 'g':
            critVal = [self.compute_gini(leftProbas), self.compute_gini(rightProbas)]
        elif critere == 'e':
            critVal = [self.compute_entropy(leftProbas), self.compute_entropy(rightProbas)]
        elif critere == 'm':
            critVal = [self.compute_misclass(leftProbas), self.compute_misclass(rightProbas)]
        return (effectifs[0] * critVal[0] + effectifs[1] * critVal[1]) / sum(effectifs)

    # Recherche la meilleure division selon le critère en argument
    def get_best_split(self, critere):
        seuil = None
        bestValue = 42.0
        bestIndex = []
        bestSeuil = []
        # NOTE: attributs symboliques d'abord, pour détecter d'éventuels regroupements.
        symbolics = list(filter(lambda x: x != self.classIndex and str(self.data.dtypes[x]) == 'object', range(0, len(self.data.columns))))
        numerics = list(filter(lambda x: x != self.classIndex and str(self.data.dtypes[x]) != 'object', range(0, len(self.data.columns))))
        symbolicFirst = symbolics + numerics
        for i0, i in enumerate(symbolicFirst):
            missings = filter(lambda x: est_manquant(x[1]), enumerate(self.data.iloc[self.indices,i]))
            missings = list( map(lambda x: self.indices[x[0]], missings) )
            nonMissings = list( set(self.indices).difference(set(missings)) )
            if i0 < len(symbolics):
                # Variable symbolique:
                classes = np.unique(self.data.iloc[nonMissings,i]) #au moins deux classes
                bestClasses = []
                bestIndices = []
                while True:
                    improvement = False
                    for c in classes:
                        if c not in bestClasses and len(bestClasses) < math.floor(len(classes) / 2):
                            newIndices = list(itertools.compress(nonMissings, self.data.iloc[nonMissings,i] == c))
                            # Placer toutes les valeurs manquantes à gauche *et* à droite:
                            gauche = bestIndices + newIndices
                            droite = list( set(self.indices).difference(set(gauche)) )
                            gauche += missings
                            if len(gauche) > 0 and len(droite) > 0:
                                value = self.eval_critere(critere, gauche, droite)
                                if value <= bestValue:
                                    improvement = True
                                    bestClasses += [c]
                                    bestIndices += newIndices
                                    if value < bestValue:
                                        bestValue = value
                                        bestIndex = [i]
                                        bestSeuil = [bestClasses]
                                    else:
                                        bestIndex += [i]
                                        bestSeuil += [bestClasses]
                    if not improvement:
                        break
            else:
                # Variable numérique:
                ordonne = self.data.iloc[nonMissings,i].argsort()
                indices = missings + [ nonMissings[j] for j in ordonne ] + missings
                M = len(missings)
                # Idée: précalculer les changements de classe et/ou de valeur.
                # Un split n'est tenté qu'aux changements de valeur, et peut parfois être évité:
                #   - si changement de classe *et* de valeur: il faut essayer
                #   - si la valeur n'a pas bougé au changement précédent: doublon multi-classe, split.
                #   - si la valeur ne bouge pas jusqu'au changement suivant: doublon multi-classe, split.
                #   - sinon: changement de valeur au sein d'une même classe => skip
                splitInfo = [ ]
                j = 0
                rightClass = self.data.iloc[indices[M],self.classIndex]
                rightValue = self.data.iloc[indices[M],i]
                while j+1 < len(nonMissings):
                    code = 0
                    leftClass = rightClass
                    leftValue = rightValue
                    rightClass = self.data.iloc[indices[M+j+1],self.classIndex]
                    rightValue = self.data.iloc[indices[M+j+1],i]
                    if leftClass != rightClass:
                        code += 1
                    if leftValue != rightValue:
                        code += 2
                    if code >= 1:
                        splitInfo += [ (j, code) ]
                    j += 1
                # Parcours tous les splits potentiels en n'effectuant que les calculs (a priori) nécessaires:
                for sx, maySplit in enumerate(splitInfo):
                    if maySplit[1] == 1:
                        continue
                    if maySplit[1] == 3 or (sx > 0 and splitInfo[sx-1][1] == 1) or (sx+1 < len(splitInfo) and splitInfo[sx+1][1] == 1):
                        firstIndexRight = M + maySplit[0] + 1
                        value = self.eval_critere(critere, indices[:firstIndexRight], indices[firstIndexRight:])
                        midPoint = (self.data.iloc[indices[firstIndexRight-1],i] + self.data.iloc[indices[firstIndexRight],i]) / 2
                        if str(self.data.dtypes[i])[0:3] == 'int':
                            midPoint = math.floor(midPoint)
                        if value < bestValue:
                            bestValue = value
                            bestIndex = [i]
                            bestSeuil = [midPoint]
                        elif value == bestValue:
                            bestIndex += [i]
                            bestSeuil += [midPoint]
        if len(bestIndex) == 0:
            return None, None
        idx = random.randint(0, len(bestIndex)-1)
        return bestIndex[idx], bestSeuil[idx]

    # Renvoie les deux sous-ensembles d'indices après division
    def do_split(self, attribut, seuil):
        gauche = []
        droit = []
        missings = [ ]
        nonMissings = [ ]
        for i in self.indices:
            valeur = self.data.iloc[i][attribut]
            # Dans un premier temps on ignore les lignes avec attribut manquant
            if est_manquant(valeur):
                missings += [i]
                continue
            else:
                nonMissings += [i]
            if str(self.data.dtypes[attribut]) != 'object':
                # Variable numérique:
                if valeur <= seuil:
                    gauche += [i]
                else:
                    droit += [i]
            else:
                # Variable symbolique: seuil = liste de str
                if valeur in seuil:
                    gauche += [i]
                else:
                    droit += [i]
        # Enfin, on répartit les lignes avec attribut manquant selon leur classe
        if len(missings) >= 1:
            nonMissings = set(self.indices).difference(set(missings))
            classesLeft, countsLeft = np.unique(self.data.iloc[gauche,self.classIndex], return_counts=True)
            classesRight, countsRight = np.unique(self.data.iloc[droit,self.classIndex], return_counts=True)
            classDictLeft = { cl[1] : countsLeft[cl[0]] for cl in enumerate(classesLeft) }
            classDictRight = { cl[1] : countsRight[cl[0]] for cl in enumerate(classesRight) }
            for m in missings:
                weightLeft = classDictLeft.get(self.data.iloc[m][self.classIndex]) or 0
                weightRight = classDictRight.get(self.data.iloc[m][self.classIndex]) or 0
                if weightRight == 0 and weightLeft == 0:
                    # Choix au hasard
                    if random.choice([0, 1]) == 0:
                        gauche += [m]
                    else:
                        droit += [m]
                elif weightRight == 0:
                    gauche += [m]
                elif weightLeft == 0:
                    droit += [m]
                else:
                    # On va à gauche avec proba #notreClasseAgauche / #notreClasse
                    if np.random.binomial(1, weightRight / (weightLeft + weightRight)) == 0:
                        gauche += [m]
                    else:
                        droit += [m]
        return gauche, droit

    # Lance l'apprentissage: fonction principale.
    # Condition d'arret: si données homogènes, ou si aucun split possible
    def learn(self, critere):
        if not self.est_homogene():
            # On recherche la meilleure coupure
            attribut, seuil = self.get_best_split(critere)
            if attribut != None:
                # On split les données selon l'attribut et le seuil de coupure
                indices_inf, indices_sup = self.do_split(attribut, seuil)
                self.attribut = attribut
                self.seuil = seuil
                self.gauche = Noeud(self.data, self.classIndex, indices_inf)
                self.droit = Noeud(self.data, self.classIndex, indices_sup)
                # alpha: différence entre l'erreur courante et celle dans les noeuds enfants
                self.alpha = self.error - self.gauche.error - self.droit.error
                self.gauche.learn(critere)
                self.droit.learn(critere)

    # Mise à jour des alpha: max(alpha des enfants) depuis la racine (post-traitement)
    def adjust_alpha(self):
        if self.est_feuille():
            return 0.0
        self.alpha = max(self.alpha, self.gauche.adjust_alpha(), self.droit.adjust_alpha())
        return self.alpha

    # Retourne toutes les valeurs alpha depuis ce noeud
    def get_alphas(self):
        if self.est_feuille():
            return []
        return [self.alpha] + self.gauche.get_alphas() + self.droit.get_alphas()

    # Retourne la classe de l'example donné en parametre
    def predict(self, exemple, alpha):
        if self.est_feuille() or alpha > self.alpha or est_manquant(exemple[self.attribut]):
            return self.get_classe()
        if str(self.data.dtypes[self.attribut]) != 'object':
            # Tout sauf symbolique
            if exemple[self.attribut] <= self.seuil:
                return self.gauche.predict(exemple, alpha)
            return self.droit.predict(exemple, alpha)
        else:
            # Variable symbolique
            if exemple[self.attribut] in self.seuil:
                return self.gauche.predict(exemple, alpha)
            return self.droit.predict(exemple, alpha)

    # Affichage textuel de l'arbre
    def affiche(self, alpha, profondeur):
        if self.est_feuille() or alpha > self.alpha:
            print(profondeur * ' ' + "* " + str(self.classe) + " " + smart_print(self.error))
        else:
            strSeuil = self.seuil
            if str(self.data.dtypes[self.attribut]) != 'object':
                strSeuil = smart_print(strSeuil)
            else:
                strSeuil = str(strSeuil)
            print(profondeur * ' ' + self.data.axes[1][self.attribut] + " " + strSeuil + " [" + smart_print(self.alpha) + "]")
            self.gauche.affiche(alpha, profondeur + 1)
            self.droit.affiche(alpha, profondeur + 1)

    # Retourne un objet GraphViz représentant le sous-graphe courant
    def to_graph(self, g, prefixe, alpha):
        if self.est_feuille() or alpha > self.alpha:
            g.node(prefixe, str(self.classe), shape='box')
        else:
            strSeuil = self.seuil
            if str(self.data.dtypes[self.attribut]) != 'object':
                strSeuil = smart_print(strSeuil)
                inf = '<='
                sup = '>'
            else:
                strSeuil = str(strSeuil)
                # Variable symbolique ('s')
                inf = '='
                sup = '!='
            g.node(prefixe, self.data.axes[1][self.attribut])
            self.gauche.to_graph(g, prefixe + "g", alpha)
            self.droit.to_graph(g, prefixe + "d", alpha)
            g.edge(prefixe, prefixe + "g", inf + str(strSeuil))
            g.edge(prefixe, prefixe + "d", sup + str(strSeuil))
        return g
