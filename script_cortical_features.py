import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, StratifiedShuffleSplit)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

n_permutations = 1000
scoring = 'neg_mean_squared_error'

# Redo the thing the data with age
df = pd.read_csv('/Users/Sofiane/Desktop/resultats_122_reseaux.csv', sep=',')
df = df[df.index.astype('str') != 'nan']
#df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
df['Z_Score_TMT_Diff_pre'] *= -1  # make more sense
networks = df.columns[:-4].tolist() + df.columns[-1:].tolist()
networks = np.array(networks)



others = df.columns[-5:-4].tolist() + df.columns[-1:].tolist()
X_ = df[others].values

"""
df1 = pd.read_csv('probability.csv', index_col=0)
df2 = pd.read_csv('proportion.csv', index_col=0)
X1 = np.hstack((df1.values, X_))
X2 = np.hstack((df2.values, X_))
"""
X_titre = df.columns[1:-1].tolist()

X=df[X_titre]
X=X.to_numpy()
do_probability = False
do_proportion = False
labels=X_titre
'''
if do_probability:
    print('Probability table')
    X = X1
    labels = list(df1.columns) + others
elif do_proportion:
    print('Proportion table')
    X = X2
    labels = list(df2.columns) + others
else:
    # baseline
    print('Baseline table')
    labels = networks
    X = df[networks].values
'''
# get the target
y = df['diff_diff'].values
plt.figure()
plt.hist(y, bins=10)

# define classifier
clf = RandomForestRegressor()  # max_depth=2, max_features=1

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
# attempt with Ridge regression
clf = RidgeCV()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('ridge:', mmae)
# attempt with GBT
clf = GradientBoostingRegressor()
mae_ = cross_val_score(clf, X, y, cv=cv, n_jobs=5,
                       scoring=scoring)
mmae = mae_.mean()
print('GBT: ', mmae)
###############################################################################
# Binary classification
# X = X[y < 1]
# y = y[y < 1]
threshold = 1.5
yb = y > threshold
scoring = 'roc_auc'
class_names = ['y<%f' % threshold, 'y>%f' % threshold,]
clf = RandomForestClassifier(max_depth=2)  # max_depth=2, max_features=1
#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)
# compute cross-val score
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print('Binary accuracy, RF: ', acc.mean())
clf.fit(X, yb)
# print(clf.feature_importances_)
print(np.array(labels)[np.argsort(clf.feature_importances_)[-5:]])
# Make an ROC curve
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
if n_permutations > 0:
    y_ = yb.copy()
    accs = []
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
        
    print(np.sum(accs > acc.mean()))
# try with single tree
clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yb, cv=cv,n_jobs=5, scoring=scoring)
print('Binary accuracy, tree: ', acc.mean())
# 
# is classification significantly good ?
# Does a tree work ?
# Feature importance
# add age
clf.fit(X, yb)
plt.figure(figsize=(8, 8))
annotations = tree.plot_tree(
    clf, feature_names=labels, class_names=class_names,
    fontsize=6, impurity=False)
plt.savefig('/tmp/tree.pdf', dpi=300)
plt.savefig('/tmp/tree.svg')
"""

#############################################################################
# Three-way classification
yt = (y > -1.5).astype(int) +  (y > 1.5).astype(int)
scoring = 'roc_auc_ovr'
class_names = ['y < -1.5', '-1.5 < y< 1.5', 'y > 1.5']

clf = RandomForestClassifier()  # max_depth=2, max_features=1

#define cross_validation scheme
cv = StratifiedShuffleSplit(n_splits=100, test_size=.25, random_state=0)

# compute cross-val score
acc = cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, RF: ', acc.mean())
clf.fit(X, yt)
print(np.array(labels)[np.argsort(clf.feature_importances_)[-5:]])

# Make an ROC curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yt, test_size=.5,
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
plt.title('Receiver operating characteristic, ternary problem ')
plt.legend(loc='lower right')
plt.savefig('/Users/sofiane/Desktop/roc_ternary.png')


if n_permutations > 0:
    y_ = yt.copy()
    accs = []
    macc = np.mean(
        cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring))
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
    print('accuracy:', macc, 'p-value, baseline',
          (1 + np.sum(accs > macc)) * 1. / n_permutations)
    macc = np.mean(
        cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring))
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
    print('accuracy:', macc, 'p-value, probability',
          (1 + np.sum(accs > macc)) * 1. / n_permutations)
    macc = np.mean(
        cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring))
    for _ in range(n_permutations):
        np.random.shuffle(y_)
        acc_ = cross_val_score(clf, X, y_, cv=cv, n_jobs=5,
                               scoring=scoring)
        accs.append(np.mean(acc_))
    print('accuracy:', macc, 'p-value, proportion',
          (1 + np.sum(accs > macc)) * 1. / n_permutations)


    
clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, tree: ', acc.mean())
clf.fit(X, yt)

plt.figure(figsize=(9, 6))
annotations = tree.plot_tree(
    clf, feature_names=labels, class_names=class_names,
    fontsize=6, impurity=False, filled=True, rounded=True)
plt.savefig('/Users/sofiane/Desktop/tree_ternary.pdf', dpi=300)
plt.savefig('/Users/sofiane/Desktop/tree_ternary.svg')

labels=X_titre

# bootstrap

