# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 23:45:16 2018

@author: lucas
"""
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43], [168, 60, 39]]

Y = ['male', 'male', 'female', 'female',
     'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'male']

# Classificadores
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Treinando os modelos com dados X e Y
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testando
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Acurácia do método DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Acurácia do método SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Acurácia do método perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Acurácia do método KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('\nO melhor método foi: {}\n'.format(classifiers[index]))

test = [168, 60, 39]

prediction = clf_tree.predict([test])
print("clf_tree classificou como: " +str(prediction))
prediction = clf_svm.predict([test])
print("clf_svm classificou como: " +str(prediction))
prediction = clf_perceptron.predict([test])
print("clf_perceptron classificou como: " +str(prediction))
prediction = clf_KNN.predict([test])
print("clf_KNN classificou como: " +str(prediction))