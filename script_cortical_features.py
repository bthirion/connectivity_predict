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

# Redo the thing the data with age
df = pd.read_csv(
    os.path.join(data_dir, 'donnees_utilisees_17_reseaux.csv'),
    sep=',')
df = df[df.index.astype('str') != 'nan']
#df.drop(labels='CorticoThalamic_4', axis=1, inplace=True)
df['Z_Score_TMT_Diff_pre'] *= -1  # make more sense
networks = df.columns[:-4].tolist() + df.columns[-1:].tolist()
networks = np.array(networks)

others = df.columns[-5:-4].tolist() + df.columns[-1:].tolist()
X_ = df[others].values

X_columns = df.columns[1:-1].tolist()

X = df[X_columns].values
labels = X_columns

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
"""
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
plt.savefig(os.path.join(write_dir, 'roc_ternary_cortical.png'))
"""
from sklearn.metrics import auc, plot_roc_curve

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, yt)):
    clf.fit(X[train], yt[train])
    y_score = clf.predict_proba(X[test])
    fpr, tpr, thresholds = metrics.roc_curve(yt[test], y_score.T[1],
                                             pos_label=1)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    # aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

aucs = acc
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic, ternary problem \n 17 networks overlap")
ax.legend(loc="lower right")
plt.savefig('/tmp/roc_ternary_17networks_overlap.png')

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
    fontsize=8, impurity=False, filled=True, rounded=True)
fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
plt.savefig(os.path.join(write_dir, 'tree_ternary_cortical.pdf'), dpi=300)
plt.savefig(os.path.join(write_dir, 'tree_ternary_cortical.svg'), dpi=300)
plt.show(block=False)


