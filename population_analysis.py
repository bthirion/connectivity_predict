"""
Population analysis of the impact of connectivity deficits on behavioral outcomes

Author: Bertrand Thirion, 2021

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, StratifiedShuffleSplit)


n_permutations = 1000
scoring = 'neg_mean_squared_error'

df = pd.read_csv('liste_patients_gliome_final_total.csv', index_col=0)

# Drop a column with only Nans
df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)

networks = df.columns[:-3]
X = df.values[:, :-3]
y = df.values[:, -2] + 0.01 * df.values[:, -1] * np.sign(df.values[:, -2])
plt.figure()
plt.hist(y, bins=10)


# define classifier
clf = RandomForestRegressor(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)

"""
# compute cross-val score
r2_ = cross_val_score(clf, X, y, cv=cv,n_jobs=5)
print(r2_.mean())

mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('rf:', mmae)

# compare to permutation distribution
maes = []
y_ = y.copy()
for _ in range(n_permutations):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring=scoring)
    maes.append(np.mean(maes_))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')
print(np.sum(maes > mmae))

# attempt with Ridge regression
from sklearn.linear_model import RidgeCV
clf = RidgeCV()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('ridge:', mmae)
for _ in range(n_permutations):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring=scoring)
    maes.append(np.mean(maes_))
print(np.sum(maes > mmae))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')

# attempt with GBT
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print(mmae)

"""

yb = y > 0
scoring = 'roc_auc'

clf = RandomForestClassifier(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print(acc.mean())

# 
# is classification significantly good ?
# Does a tree work ?
# Feature importance
# 

plt.show(block=False)
