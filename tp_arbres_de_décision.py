    #%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import graphviz
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
# Ajouter un chemin d'accès à sys.path
sys.path.append(r"C:\Users\Abdoulaye Diop\Desktop\TP2_AS\TP - arbres-20221006")

# Maintenant, vous pouvez importer des modules de ce chemin d'accès
import tp_arbres_source

from sklearn import tree, datasets
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                            rand_checkers, rand_clown,
                            plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
        'font.size': 12,
        'legend.fontsize': 12,
        'text.usetex': False,
        'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114  # XXX : change
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
#data4 = rand_checkers(n1, n2, n3, n4, sigma)

#%%
############################################################################
# Displaying labeled data
############################################################################

#%%
############################################
# ARBRES
############################################


# Q2. 
# Créer deux objets 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.
# Créer un arbre de décision avec l'indice de Gini comme critère
dt_gini = DecisionTreeClassifier(criterion='gini')

# Créer un arbre de décision avec l'entropie comme critère
dt_entropie = DecisionTreeClassifier(criterion='entropy')

# dt_entropy = TODO
# dt_gini = TODO

# Effectuer la classification d'un jeu de données simulées avec rand_checkers des échantillons de
# taille n = 456 (attention à bien équilibrer les classes)
n=114 # n*4=456
data = rand_checkers(n, n, n, n, sigma)
# data = TODO
n_samples = len(data)

# X = TODO
# Y = TODO and careful with the type (cast to int)

X = data[:, :2]
Y = data[:, 2].astype(int)

# Entrainement des données 
dt_gini.fit(X, Y)
dt_entropie.fit(X, Y)

print("Gini criterion")
print(dt_gini.get_params())
print(dt_gini.score(X, Y))
#%%
# Afficher les scores en fonction du paramètre max_depth
# on choisit comme le pronfondeur de l'arbre est égale à 12
dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

plt.figure(figsize=(15, 10))
X = data[:, :2]
Y = data[:, 2].astype(int)

for i in range(dmax):
    # dt_entropy = ... TODO
     dt_entropy=DecisionTreeClassifier(criterion='entropy',max_depth=i+1)
     dt_entropy.fit(X, Y)
     scores_entropy[i] = dt_entropy.score(X, Y)

    # dt_gini = ... TODO
    # ...
    dt_gini=DecisionTreeClassifier(criterion='gini',max_depth=i+1)
    dt_gini.fit(X,Y)
    scores_gini[i] = dt_gini.score(X, Y)

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X, Y, step=50, samples=False)


    plt.draw()

plt.figure()
plt.plot(np.arange(1, dmax + 1), scores_entropy)
plt.plot(np.arange(1, dmax + 1), scores_gini)
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
#plt.draw()
print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

# Question 3
#%%
# Q3 Afficher la classification obtenue en utilisant la profondeur qui minimise le pourcentage d’erreurs
# obtenues avec l’entropie

# dt_entropy.max_depth = ... TODO
# recuperation de l'indice(profondeur maximale qui maximise l'score)
dt_entropy.max_depth = np.argmax(scores_entropy) + 1
dt_entropy.fit(X, Y)

frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=100)
plt.title("Best frontier with entropy criterion")
plt.draw()
print("Best scores with entropy criterion: ", dt_entropy.score(X, Y))

# Question 4 

#%%
# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed
# Voir https://scikit-learn.org/stable/modules/tree.html#classification

# TODO
# le graphe ainsi  obtenu est noté arbre dans le fichier DATA
tree.plot_tree(dt_entropy)
import graphviz 
dot_data = tree.export_graphviz(dt_entropy, out_file=None) 
graph = graphviz.Source(dot_data)


# Question 5

#%%
# Q5 :  Génération d'une base de test
n_test=40
data_test =rand_checkers(n_test, n_test, n_test, n_test, sigma)
X_test= data_test[:, :2]
Y_test= data_test[:, 2].astype(int) 
dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(15, 10))

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i + 1)
    dt_entropy.fit(X, Y)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i + 1)
    dt_gini.fit(X, Y)
    scores_gini[i] = dt_gini.score(X_test, Y_test)

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=50)
plt.draw()

#%%
    plt.figure()
# plt.plot(...)  # TODO
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.title("Testing error")
print(scores_entropy)



# Question complete

n_test=40
data_test =rand_checkers(n_test, n_test, n_test, n_test, sigma)
X_test= data_test[:, :2]
Y_test= data_test[:, 2].astype(int) 
dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(15, 10))

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i + 1)
    dt_entropy.fit(X, Y)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i + 1)
    dt_gini.fit(X, Y)
    scores_gini[i] = dt_gini.score(X_test, Y_test)

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X, Y, step=50)
plt.draw()

#%%
plt.figure()
plt.plot(np.arange(1, dmax + 1), scores_entropy)
plt.plot(np.arange(1, dmax + 1), scores_gini)

plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.title("Testing error")
print(scores_entropy)


# Question 6

digits = datasets.load_digits()



n_samples = len(digits.data)
# use test_train_split rather.

X = digits.data[:n_samples // 2]  # digits.images.reshape((n_samples, -1))
# Y = digits.target[:n_samples // 2]
X_test= data_test[:, :2]
Y_test= data_test[:, 2].astype(int) 

# Question 7


ent_sc = []
gini_sc= []

for depth in range(1, 13):
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    acc = cross_val_score(dt, X_train, Y_train, cv=5, n_jobs=-1)
    ent_sc.mean()

    dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth)
    acc = cross_val_score(dt, X_train, Y_train, cv=5, n_jobs=-1)
    gini_sc.mean()


plt.figure(figsize=(10, 6))
plt.plot(range(1, 12), ent_sc, label="entropy")
plt.plot(range(1, 12), gini_sc, label="gini")
plt.xlabel('Pronfondeur')
plt.ylabel("Score")
plt.legend()
plt.title(""" Cross Validation K_Fold avec K=5""")
plt.show()

dt_cv = tree.DecisionTreeClassifier(criterion='entropy', max_depth=np.argmin(ent_sc) + 1) 
dt_cv.fit(X_test, Y_test)
