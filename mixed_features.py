"""
This script does the same as  population_analysis.py but with mixed features:
cortical and white matter features for cortico-cortical networks only.

Author: Bertrand Thirion, 2021
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, StratifiedShuffleSplit)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

data_dir = '.'
write_dir = '/tmp'
n_permutations = 0

# Read the cortical lesion data 
df = pd.read_csv(
    os.path.join(data_dir, 'donnees_utilisees_17_reseaux.csv'),
    sep=',')
df = df[df.index.astype('str') != 'nan']
df['Z_Score_TMT_Diff_pre'] *= -1  # make more sense
X_columns = df.columns[1:-1].tolist()
X2 = df[X_columns].values

#then additionally read the network lesion data
df1 = pd.read_csv('liste_patients_gliome_final_total_avec_AGE_NSC.csv', sep=',',
                 index_col=0)

df1 = df1[df.index.astype('str') != 'nan']
df1.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
df1['Z_Score_TMT_Diff_pre'] *= -1  # make more sense
networks = df1.columns.tolist()
# keep only cortico-cortical newtorks
networks = np.array([x for x in networks if 'Cortical' in x])
X1 = df1[networks].values

# Check that the two table display the same participants
subjects = [x.split('_')[-1] for x in df1.index]
for s1, s2 in zip(subjects, list(df['Patient'])):
    if s1 != s2:
        print(s1, s2)

# Merge the two type of features (hence, "mixed" features)
X = np.hstack((X1, X2))
labels = np.hstack((networks, X_columns))

# get the target
y = df['diff_diff'].values
plt.figure()
plt.hist(y, bins=10)

# define classifier
clf = RandomForestRegressor()  # max_depth=2, max_features=1

#define cross_validation scheme
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)


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
plt.savefig(os.path.join(write_dir, 'roc_ternary_mixed.png'))


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

    
clf = DecisionTreeClassifier(max_depth=3)
acc = cross_val_score(clf, X, yt, cv=cv,n_jobs=5, scoring=scoring)
print('Ternary accuracy, tree: ', acc.mean())
clf.fit(X, yt)

fig = plt.figure(figsize=(9, 4))
ax = plt.subplot(111)
annotations = tree.plot_tree(
    clf, feature_names=labels, class_names=class_names, ax=ax,
    fontsize=6, impurity=False, filled=True, rounded=True)
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
plt.savefig(os.path.join(write_dir, 'tree_ternary_mixed.pdf'), dpi=300)
plt.savefig(os.path.join(write_dir, 'tree_ternary_mixed.svg'), dpi=300)
plt.show(block=False)


