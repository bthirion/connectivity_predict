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


n_permutations = 100
scoring = 'neg_mean_squared_error'

"""
df = pd.read_csv('liste_patients_gliome_final_total.csv', index_col=0)
# Drop a column with only Nans
df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
networks = df.columns[:-3]
X = df.values[:, :-3]
y = df.values[:, -2] + 0.01 * df.values[:, -1] * np.sign(df.values[:, -2])
X_ = X.copy()
"""

# Redo the thing the data with age
df = pd.read_csv('liste_patients_gliome_final_total_avec_AGE_NSC.csv',
                 index_col=0)
df = df[df.index.astype('str') != 'nan']
df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
networks = df.columns[:-4].tolist() + df.columns[-1:].tolist()
networks = np.array(networks)

X = np.hstack((df.values[:, :-4], df.values[:, -1:]))
y = df.values[:, -3]
# what about NSC ?


plt.figure()
plt.hist(y, bins=10)


# define classifier
clf = RandomForestRegressor(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)


# compute cross-val score
r2_ = cross_val_score(clf, X, y, cv=cv,n_jobs=5)
print(r2_.mean())


mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('rf:', mmae)

"""
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
"""

# attempt with Ridge regression
from sklearn.linear_model import RidgeCV
clf = RidgeCV()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('ridge:', mmae)
"""
for _ in range(n_permutations):
    np.random.shuffle(y_)
    maes_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                       scoring=scoring)
    maes.append(np.mean(maes_))
print(np.sum(maes > mmae))

plt.figure()
plt.hist(maes, bins=10)
plt.plot(mmae, 0, '*r')
"""

# attempt with GBT
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print(mmae)


X = X[y < 1]
y = y[y < 1]

threshold = -1.5
yb = y > threshold
scoring = 'roc_auc'

clf = RandomForestClassifier(max_depth=2)  # max_depth=2, max_features=1

#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print(acc.mean())

clf.fit(X, yb)
print(clf.feature_importances_)
print(np.array(networks)[np.argsort(clf.feature_importances_)[-5:]])

# plt.figure()
# plt.scatter(X[:, networks == 'Z_Score_TMT_Diff_pre'], y)

# Make an ROC curve

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yb, test_size=.5,
                                                    random_state=0)

y_score = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score.T[0], pos_label=0)
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % np.mean(acc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.savefig('/tmp/roc.png')

do_permutation = True
if do_permutation:
    # permutation test:
    n_permutations = 1000
    y_ = yb.copy()
    accs = []
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
        
    print(np.sum(accs > acc.mean()))


# try with single tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print(acc.mean())


# 
# is classification significantly good ?
# Does a tree work ?
# Feature importance
# add age
clf.fit(X, yb)
from sklearn import tree
class_names = ['y<%f' % threshold, 'y>%f' % threshold,]
plt.figure(figsize=(8, 8))
annotations = tree.plot_tree(
    clf, feature_names=networks, class_names=class_names,
    fontsize=6, impurity=False)
plt.savefig('/tmp/tree.pdf', dpi=300)
# plt.savefig('/tmp/tree.png', dpi=300)
plt.savefig('/tmp/tree.svg')





plt.show(block=False)
