"""
This script reads the summaries of the imaging data to create
compact features for prediction.

Author: Bertrand Thirion, 2020
"""

import pandas as pd
import numpy as np

# replace with wherever you put that file
csv_file ='/Users/lebbe/Downloads/re/all_connections.csv'
csv_file_behavior = '/Users/lebbe/Downloads/behavior.csv'

#directory where figures are saved
mypath = '/Users/lebbe/Downloads/'

# this is the whole dataframe
all_connections = pd.read_csv(csv_file, index_col=0)
#this is behavioral data ( 1 indicates the presence of perseveration)
behavior = pd.read_csv(csv_file_behavior, index_col=0)
# get the pathway names
connection_ids = all_connections.columns[2:] # discard subjectID and TrackID
unique_ids = np.unique([c.split('_')[0] + '_'  + c.split('_')[1]
                        for c in connection_ids])

##############################################################################
# aggregate connectivity values from left, right etc.
# by summing them to a unique values
aggregated_connectivity = {}
for id_ in unique_ids:
    relevant_ids = [c for c in connection_ids
                    if c.startswith(id_ + '_') or c == id_]
    total_ids = [c for c in relevant_ids if c.endswith('total')]
    partial_ids = [c for c in relevant_ids if not c.endswith('total')]
    aggregated_connectivity[id_] = all_connections[partial_ids].sum(1).values
    aggregated_connectivity[id_ + '_total'] = all_connections[total_ids]\
                                              .sum(1).values

# make a dataframe from it
aggregated_connectivity = pd.DataFrame(aggregated_connectivity)
# add the missing columns
aggregated_connectivity['subjectID'] = all_connections['subjectID']

##############################################################################
# Average per subject across HCP counts
unique_subjects = all_connections['subjectID'].unique()
average_connectivity = {}
for subject in unique_subjects:
    x = aggregated_connectivity[
        aggregated_connectivity.subjectID == subject]
    average_connectivity[subject] = aggregated_connectivity[
        aggregated_connectivity.subjectID == subject].sum(0)
    # todo: more sophisticated averaging scheme to deal with low values
    # since those values are unreliable
 
# make a dataFrame from it
average_connectivity = pd.DataFrame(average_connectivity).T

#add subject ID
average_connectivity['subjectID'] = unique_subjects

##############################################################################
#other methods to deal with low values :
# method BIS : sum of ratios
# 1) for each subject make a sum of ratios obtained in each HCP connectome
# 2) transform ratios with sigmoid function

unique_subjects = all_connections['subjectID'].unique()
ratios_connectivity_bis_HCP = {}
sumratios_connectivity_bis_subject = {}

for id_ in unique_ids:
    ratios_connectivity_bis_HCP[id_] = aggregated_connectivity[id_] / (
        1. + aggregated_connectivity[id_ + '_total'])
ratios_connectivity_bis_HCP = pd.DataFrame(ratios_connectivity_bis_HCP)
ratios_connectivity_bis_HCP['subjectID'] = all_connections['subjectID']

for subject in unique_subjects:
    x = ratios_connectivity_bis_HCP[
        ratios_connectivity_bis_HCP.subjectID == subject]
    sumratios_connectivity_bis_subject[subject] = ratios_connectivity_bis_HCP[
        ratios_connectivity_bis_HCP.subjectID == subject].sum(0)
sumratios_connectivity_bis_subject = pd.DataFrame(
    sumratios_connectivity_bis_subject).T
sumratios_connectivity_bis_subject_transformed = 1 / (1 + np.exp(np.asarray(
    - sumratios_connectivity_bis_subject.drop(['subjectID'], axis=1),
    dtype=float)))
sumratios_connectivity_bis_subject_transformed = pd.DataFrame(
    sumratios_connectivity_bis_subject_transformed)
sumratios_connectivity_bis_subject_transformed['subjectID'] = unique_subjects
ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
ANTS_sumratios_connectivity_bis_subject_transformed = (
    sumratios_connectivity_bis_subject_transformed[
        sumratios_connectivity_bis_subject_transformed.subjectID.isin(ANTS)])

##############################################################################
# Keep only ANTS subjects

# ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]

ANTS_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(ANTS)]

# Todo: do the same with FSL_connectivity
FSL = [subject for subject in unique_subjects if subject.endswith('FSL')]
FSL_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(FSL)]

##############################################################################
# finally compute the  partial/total ratio in each subject
ANTS_ratio = {}
for id_ in unique_ids:
    ANTS_ratio[id_] = ANTS_connectivity[id_] / (
        1. + ANTS_connectivity[id_ + '_total'])

# make a DataFrame from it
ANTS_ratio = pd.DataFrame(ANTS_ratio)
ANTS_ratio.name = 'ANTS_ratio'
#transform data with sigmoid function 
ANTS_ratio_transformed = 1 / (1 + np.exp(np.asarray(- ANTS_ratio,dtype=float)))
ANTS_ratio_transformed = pd.DataFrame(ANTS_ratio_transformed)
ANTS_ratio_transformed.name = 'ANTS_ratio_transformed'
# ANTS_ratio supposeldy contains some data that are ready for machine learning
# do the same with FSL_connectivity
FSL_ratio = {}
for id_ in unique_ids:
    FSL_ratio[id_] = FSL_connectivity[id_] / (
        1. + FSL_connectivity[id_+'_total'])

# make a DataFrame from it : 
FSL_ratio = pd.DataFrame(FSL_ratio)
FSL_ratio.name = 'FSL_ratio'
#transform data with sigmoid function 
FSL_ratio_transformed = 1 / (1 + np.exp(np.asarray(- FSL_ratio,dtype=float)))
FSL_ratio_transformed = pd.DataFrame(FSL_ratio_transformed)
FSL_ratio_transformed.name = 'FSL_ratio_transformed'
##############################################################################
#plot ANTS ratio (corticocortical, corticostriatal, corticothalamic)
# import plotting libraries and pi

import matplotlib.pyplot as plt
from math import pi

# define radar plot function

def make_spider(row, title, liste, color, j, k):
    N = len(categories)
    # define the angle for each variable
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]   
    # initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    # Add behavior
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='blue',
            fontsize=12)
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='red', fontsize=12)
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='purple', fontsize=12)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1],
               [i.partition('_')[2] for i in categories],
               color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9],
               ["0.1", "0.3", "0.5", "0.7", "0.9"], color="grey", size=7)
    plt.ylim(0, 1)
     
    # Ind1
    values = liste.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:-1], values[j:k], color=color, linewidth=2,
            linestyle='solid')
    ax.fill(angles[0:-1], values[j:k], color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color='grey', y=1.1)

def loop_to_plot(liste, j, k):
    # initialize the figure
    my_dpi = 96
    plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
     
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(liste.index))
     
    # Give a name to big figure
    plt.gcf().text(0.9, 0.9, 
                   str(categories[0].partition('_')[0]) + '_' 
                    + liste.name, fontsize=40)
    # Loop to plot
    for row in range(0,len(liste.index)):
        make_spider(row=row, liste = liste, title=liste.index[row], 
                    color=my_palette(row), j=j, k=k)
    # save figure 
    plt.savefig(mypath + str(categories[0].partition('_')[0]) + '_' 
            + liste.name + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, frameon=None, metadata=None)

# Create one figure with all individuals for each level 
# (CorticoCortical, CorticoStriatal, CorticoThalamic)

# First figure : ANTS CorticoCortical (available in dataset =17)
j = 0
k = 17
liste=ANTS_ratio
categories = list(liste)[j:k] 
loop_to_plot (liste=liste, j=j, k=k)
# Second figure ANTS CorticoStriatal (available in dataset n=15)
j = 17
k = 32
categories = list(liste)[j:k] 
loop_to_plot (liste=liste, j=j, k=k) 
# Third figure ANTS CorticoThalamic networks (available in dataset n=14)
j = 32
k = 46
liste=ANTS_ratio
categories = list(liste)[j:k] 
loop_to_plot (liste=liste, j=j, k=k)



##############################################################################
# predict presence of verbal perseveration early postop by applying different supervised learning analysis
from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, KFold, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#select disconnectome data : here we chose the first method of averaging

X = ANTS_ratio.iloc[:, 0:46]
y = behavior.iloc[:, 3]

# remove subject with nan value from both datasets (here the second line)
X = X.drop(X.index[1])
y = y.drop(y.index[1])

####SVC rbf
nested_score_rbf=np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
svm = SVC(kernel="rbf") 
# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
nested_score_rbf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of {:6f} with std. dev. of {:6f}.'.format(nested_score_rbf.mean(),nested_score_rbf.std()))

###SVC linear
nested_score_linear=np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
svm = SVC(kernel="linear") 
# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
nested_score_linear = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(nested_score_linear.mean(),nested_score_linear.std()))

###random forest
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : [2,4,6,8,10,12,14,16,18,20]}
rf= RandomForestClassifier(random_state=0)

# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
nested_score_rf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(nested_score_rf.mean(),nested_score_rf.std()))

###Logistic Regression
nested_score_lr = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100]}
lr= LogisticRegression(random_state=0)

# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=lr, param_grid=p_grid, cv=inner_cv)
nested_score_lr = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(nested_score_lr.mean(),nested_score_lr.std()))