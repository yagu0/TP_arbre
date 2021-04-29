import math

# pandas marque les valeurs manquantes avec NaN
def est_manquant(arg):
    return isinstance(arg, float) and math.isnan(arg)

# Affichage avec exposant si le nombre est proche de 0 ou grand en valeur absolue
def smart_print(arg):
    if arg < 100 and arg > -100 and abs(10*arg) >= 1.0:
        return str(round(arg, 3))
    return '{:0.2e}'.format(arg)
