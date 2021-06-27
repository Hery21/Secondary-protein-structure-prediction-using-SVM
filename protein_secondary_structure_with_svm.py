import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

lineList = [line.rstrip('\n') for line in open('RS126.data')]

prm,skn = lineList[::2], lineList[1::2]

for i in range(len(prm)-1):
  if (len(prm[i]) != len(skn[i])):
    prm.pop(i)
    skn.pop(i)

def split(word):
  return [char for char in word]

prm2 = []
skn2 = []

for i in range(len(prm)-1):
  prm2.append(split(prm[i]))
  skn2.append(split(skn[i]))

prm2 = np.concatenate(prm2)

skn2 = np.concatenate(skn2)

val1 = array(prm2)
val2 = array(skn2)
le = LabelEncoder()
ie = le.fit_transform(val1)
skn3 = le.fit_transform(val2)
ohe = OneHotEncoder(sparse=False)
ie = ie.reshape(len(ie), 1)
prm3 = ohe.fit_transform(ie)

print("=======================Without Gridsearch=======================")
X_train, X_test, y_train, y_test = train_test_split(prm3, skn3, test_size = 0.20)
svc = SVC()
svc.fit(X_train, y_train)

print("=======================With Gridsearch=======================")
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
              
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)