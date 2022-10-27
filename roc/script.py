"""
Analysis of the reprise data
"""
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


def convert_number(x):
    """ """
    return np.array([float(t.replace(',', '.')) for t in x])
    
df = pd.read_csv('Roc_curve_reprise.csv')
y = df['Non reprise'].values
x1 = convert_number(df['Fluence sémantique (animaux), Rapport score post/pré (%)'])
x2 = convert_number(df['Fluence sémantique (animaux), delta Z-score post-pré'])
x3 = convert_number(df['Fluence phonémique, Rapport score post/pré (%)'])
x4 = convert_number(df['Fluence phonémique, delta Z-score'])

n_perm = 100000
names = df.columns[2:]

for i, (x, name) in enumerate(zip([x1, x2, x3, x4], names)):
    fpr, tpr, thresholds = roc_curve(y, -x)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - %s" % name)
    plt.legend(loc="lower right")
    plt.savefig('roc_%d.png' % i)
    y_ = y.copy()
    roc_aucs = np.zeros(n_perm)
    for i in range(n_perm):
        np.random.shuffle(y_)
        fpr, tpr, thresholds = roc_curve(y_, -x)
        roc_aucs[i] = auc(fpr, tpr) 

    p_value = np.sum(roc_aucs > roc_auc) / len(roc_aucs)
    print(name, 'AUC: %1.3f' % roc_auc, 'Non-parametric-pvalue: %1.3f' % p_value)
        
plt.show()
