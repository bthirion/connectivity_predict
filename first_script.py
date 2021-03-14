#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:34:15 2020

@author: lebbe
"""

"""
This script reads the summaries of the imaging data to create
compact features for prediction.
Author: Bertrand Thirion, 2020
"""
#import libraries
import os
import pandas as pd
import numpy as np
import nibabel as nb
from scipy import stats
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from math import pi

from nilearn import plotting

from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, KFold, GridSearchCV,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor)
from sklearn import tree
from sklearn.tree import (DecisionTreeClassifier,export_text)
from sklearn.datasets import make_regression


# replace with wherever you put that file
if 1:
    data_dir = '/Users/lebbe/Downloads/'
else:
    data_dir = os.getcwd()

# setting it only once to better adapt to local configuration 
csv_file = os.path.join(data_dir, 'all_connections.csv')
csv_file_supp = os.path.join(data_dir, 'suppl_connections.csv')
csv_file_behavior = os.path.join(data_dir, 'behavior_8_07022021.csv')

# directory where figures are saved
write_dir = data_dir
image_dir = '/Users/lebbe/Dropbox/test_nilearn/analyses_septembre2020/EF_plot/' 
# this is the whole dataframe
all_connections = pd.read_csv(csv_file, index_col=0)
#this is behavioral data ( 1 indicates the presence of perseveration, 2 
# indicates one patient with intriguing postoperative behavior)
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
# Separate into ANTS and FSL subjects

ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
ANTS_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(ANTS)]

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

# transform data with sigmoid function 
ANTS_ratio_transformed = 1 / (1 + np.exp(np.asarray(- ANTS_ratio,dtype=float)))
ANTS_ratio_transformed = pd.DataFrame(ANTS_ratio_transformed)
ANTS_ratio_transformed.name = 'ANTS_ratio_transformed'

# ANTS_ratio supposeldy contains some data that are ready for machine learning
# do the same with FSL_connectivity
FSL_ratio = {}
for id_ in unique_ids:
    FSL_ratio[id_] = FSL_connectivity[id_] / (
        1. + FSL_connectivity[id_+'_total'])

# Make a DataFrame from it : 
FSL_ratio = pd.DataFrame(FSL_ratio)
FSL_ratio.name = 'FSL_ratio'

# Transform data with sigmoid function 
FSL_ratio_transformed = 1 / (1 + np.exp(np.asarray(- FSL_ratio, dtype=float)))
FSL_ratio_transformed = pd.DataFrame(FSL_ratio_transformed)
FSL_ratio_transformed.name = 'FSL_ratio_transformed'

##############################################################################
#add supplementary patients
supp_ratio = pd.read_csv(csv_file_supp, index_col=0, sep=";")
ANTS_ratio=pd.concat([ANTS_ratio, supp_ratio])


##############################################################################
# LISTS ACCORDING TO BEHAVIOR, LISTS OF NETWORKS
# change name of columns with abbreviations to make radar plots more readable
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoThalamic", "CT")
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoStriatal", "CS")
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoCortical", "CC")

# remove networks for which data are not reliable
unreliable_networks = ["CC_1", "CC_2", "CC_3", "CC_11", "CS_1", "CS_2",
                       "CS_5", "CS_6", "CS_14", "CT_1", "CT_2", "CT_3",
                       "CT_4", "CT_5", "CT_6", "CT_9", "CT_10", "CT_11",
                       "CT_13", "CT_15"]
ANTS_ratio_clean = ANTS_ratio.drop(unreliable_networks, axis=1)

# change ANTS_ratio_clean index so that behavior and ANTS_ratio have the same
# index
ANTS_ratio_clean.index = behavior.index
#concatenate ANTS_ratio_clean and behavior
ANTS_ratio_clean = pd.concat([ANTS_ratio_clean, behavior],axis=1)


# defines group lists in which subjects are grouped according to their 
#intraoperative and postoperative behavior
#cannot make a loop for this)
A = ANTS_ratio_clean.loc[ANTS_ratio_clean['eVP'] == 0]
A.name = "A"
B = ANTS_ratio_clean.loc[ANTS_ratio_clean['eVP'] == 1]
B.name = "B"
C = ANTS_ratio_clean.loc[ANTS_ratio_clean['lVP'] == 0]
C.name = "C"
D = ANTS_ratio_clean.loc[ANTS_ratio_clean['lVP'] >= 1]
D.name = "D"
DD=D[D.eVP!=0]
DD.name = "DD"
E = ANTS_ratio_clean.loc[ANTS_ratio_clean['siVP'] == 0]
E.name = "E"
F = ANTS_ratio_clean.loc[ANTS_ratio_clean['siVP'] == 1]
F.name = "F"
G = F.loc[F['eVP'] == 0]
G.name = "G"
H = F.loc[F['eVP'] == 1]
H.name = "H"
I = H.loc[H['lVP'] == 0]
I.name = "I"
J = H.loc[H['lVP'] == 1]
J.name = "J"
K = ANTS_ratio_clean.loc[ANTS_ratio_clean['lVP'] == 0]
K.name = "K"
L = E.loc[E['lVP'] == 1]
L.name = "L"


#define group list ( J is removed because is empty)
group_list = [A, B, C, D, DD, E, F, G, H, I, K, L]
group_means={}
for group in group_list:  
    group_means[group.name] = group.mean(skipna=True)[:26]
group_means = pd.DataFrame(group_means)
group_means.to_excel(data_dir + '/group_means.xlsx')

neuro_columns = [
    'CC_10', 'CC_12', 'CC_13', 'CC_14', 'CC_15', 'CC_16', 'CC_17', 'CC_4',
    'CC_5', 'CC_6', 'CC_7', 'CC_8', 'CC_9', 'CS_10', 'CS_12', 'CS_13',
    'CS_16', 'CS_17', 'CS_3', 'CS_4', 'CS_7', 'CS_8', 'CS_9', 'CT_16',
    'CT_7', 'CT_8']
    
# defines radar_lists in which patients are grouped according to radar plot 
#pattern ( PROBABLY NEED TO REMOVE THIS PART OF THE SCRIPT)

# dixdixsept =  ANTS_ratio_clean.loc[ANTS_ratio_clean['dixdixsept'] == 1]
# dixdixsept.name = "dixdixsept"
# small =  ANTS_ratio_clean.loc[ANTS_ratio_clean['small'] == 1]
# small.name = "small"
# neuf =  ANTS_ratio_clean.loc[ANTS_ratio_clean['neuf'] == 1]
# neuf.name = "neuf"
# quatre =  ANTS_ratio_clean.loc[ANTS_ratio_clean['quatre'] == 1]
# quatre.name = "quatre"
# sept=  ANTS_ratio_clean.loc[ANTS_ratio_clean['sept'] == 1]
# sept.name = "sept"
# douze =  ANTS_ratio_clean.loc[ANTS_ratio_clean['douze'] == 1]
# douze.name = "douze"

# radar_list = [dixdixsept,small,neuf,quatre,sept,douze]

##############################################################################
# CALCULATE VOLUME OF RESECTION CAVITY (in MNI space) FROM .NII  FILE
for liste in group_list : 
    for i in liste.index : 
        INPUT = image_dir + liste.loc[i, "NaNme_image"]
    
    # Load data
        nii = nb.load(INPUT)
        img = nii.get_fdata()
    
    # Get voxel dimensions
        voxel_dims = (nii.header["pixdim"])[1:4]
        # print("Voxel dimensions:")
        # print("  x = {} mm".format(voxel_dims[0]))
        # print("  y = {} mm".format(voxel_dims[1]))
        # print("  z = {} mm".format(voxel_dims[2]))
    
    # Compute volume
        nonzero_voxel_count = np.count_nonzero(img)
        voxel_volume = np.prod(voxel_dims)
        nonzero_voxel_volume = nonzero_voxel_count * voxel_volume
        liste.loc[i, "volume_lesion"]= nonzero_voxel_volume

##############################################################################
# Plotting functions

#create scatter plot for each level ( CC16, CS 16, CT 16) displaying
# disconnection index in patients with or without lVP
#WORK TO DO : find a way to display name of network on graph
def scatter_network(network, liste, behavior) :
    for level in network : 
        my_dpi = 300
        fig, ax = plt.subplots(figsize=(3000 / my_dpi, 7000/ my_dpi), dpi=my_dpi, 
                               constrained_layout=True)
        ax.scatter(liste[behavior], liste[level], s=5)
        for x,y,z in zip(liste[behavior],liste[level],liste.index):
            label = f"({x},{y})"
            plt.annotate(z.split('_')[1],(x,y),textcoords="offset points",
                         xytext=(10,10), size=5,
                         ha='center',bbox=dict(boxstyle = 'round,pad=0.1', 
                                               fc = 'yellow',alpha = 0.5))
        plt.savefig(write_dir+level+"_scatter.png")
        
#create 3D plot to show relation between parameters in different lists

def tridim_plot (listes,x,y,z,name) : 
    for liste in listes : 
        fig=plt.figure()
        ax= fig.add_subplot(111, projection='3d')
    
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.title(str(liste.name))
        ax.scatter(liste[x],liste[y],liste[z])
        plt.show()
        plt.savefig(write_dir+liste.name+ name + ".png")
    
#plots showing disconnection of CC16, lesion volume and postoperative behavior
def relplot (liste,network,behavior)
    ax= sns.relplot(1,liste[network], hue = liste[behavior], 
                    size = liste["volume_lesion"]/1000)
    plt.savefig(write_dir+ network +"_scatter.png")


def plot_smoothed_distibution(liste,hue,network):
    sns.displot( liste,x=pd.to_numeric(liste[network]), hue=hue, kind="kde",
                multiple="stack")
    plt.savefig(write_dir + '_' + network + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)

def plot_normalized_histogram(liste,hue,network,bins):
    my_dpi = 300
    fig = plt.figure(figsize=(20000/ my_dpi, 20000 / my_dpi), dpi=my_dpi)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Distribution of disconnection index for each network ")
    sns.displot( liste,x=pd.to_numeric(liste[network]), hue=hue)
    plt.savefig(write_dir + '_' + network + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)

def make_spider2(categories, row, title, liste, color, j, k,ax):
    
    n_categories = len(categories)
    # define the angle for each variable
    angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
    angles += angles[:1]   
    # initialize the spider plot
#    ax2 = spider.add_axes( polar=True )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i for i in categories], color='black', size=6)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9],
               ["0.1", "0.3", "0.5", "0.7", "0.9"], color="grey", size=7)
    plt.ylim(0, 1)
     
    # Ind1
    values = liste.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0: -1], values[j: k], color=color, linewidth=2,
            linestyle='solid')
    ax.fill(angles[0: -1], values[j: k], color=color, alpha=0.4)
    # add information regarding behavior
    # add information regarding siVP



    # Add a title
    plt.title(title, size=8, color='grey', y=1.3)

#    plt.subplots_adjust(left=0.125,
#                        bottom=0.1, 
#                        right=0.9, 
#                        top=0.9, 
#                        wspace=0.1, 
#                        hspace=0.4)


def loop_to_plot2(categories, liste, j, k):
    # initialize the figure
    my_dpi = 300
    fig = plt.figure(figsize=(9000 / my_dpi, 9000 / my_dpi), dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(liste.index))

    # Give a name to big figure
#    plt.gcf().text(0.5, 0.9, liste.name, fontsize=20)
   #the numbers in gridspec
    outer = gridspec.GridSpec(8, 5, wspace=0.2, hspace=0.3)

    # Loop to plot
    for row in range(0,len(liste.index)):
        inner = gridspec.GridSpecFromSubplotSpec(2,1,
                    subplot_spec=outer[row], wspace=0.1, hspace=0.1)
        ax = fig.add_subplot(inner[0],polar=True)
        make_spider2(categories, row=row, liste = liste,
                     title=liste.name_anonym[row], 
                     color=my_palette(row), j=j, k=k, ax=ax)
        ax2 = fig.add_subplot(inner[1])
        plotting.plot_roi(image_dir + liste["name_image"][row],axes=ax2)
        props_red = dict(boxstyle='circle', facecolor='red', alpha=0.5)
        props_green = dict(boxstyle='circle', facecolor='green', alpha=0.5)
        props_grey = dict(boxstyle='circle', facecolor='grey', alpha=0.5)
        bhv = [(28, "lVP", 0.90), (30, "siVP",0.70), (31, "eVP", 0.80)]
        for b, h, v in bhv :
            if liste.values[row,b] == 1 : 
                ax2.text(v, 1, h + '+', transform=ax2.transAxes, fontsize=4, 
                        verticalalignment = "top", bbox=props_red)
            elif liste.values[row, b]==0:
                ax2.text(v, 1, h + '-', transform=ax2.transAxes, fontsize=4, 
                        verticalalignment = "top", bbox=props_green)
            else :            
                ax2.text(v, 1, h + '?', transform=ax2.transAxes, fontsize=4, 
                        verticalalignment = "top", bbox=props_grey)
    # save figure 
    plt.savefig(write_dir + str(categories[0].partition('_')[0]) + '_' 
            + liste.name + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)

##############################################################################
### PLOTS
#radar plot organized by behavior
for letter in group_list: 
    # define all columns with disconnection ratios in dataframe
    j = 0 
    k = 26
    liste = letter
    categories = list(liste)[j:k] 
    loop_to_plot2(categories, liste=liste, j=j, k=k)
#radar plots organized by pattern of disconnection
for letter in radar_list: 
    # define all columns with disconnection ratios in dataframe
    j = 0 
    k = 26
    liste = letter
    categories = list(liste)[j:k] 
    loop_to_plot2(categories, liste=liste, j=j, k=k)

# create scatter plot to show distribution of disconnection index at each 
#level( one plot per level) according to behavior
    network = ["CC_16","CS_16", "CT_16"]
    behavior ="lVP"
    liste = CD
    scatter_network(network=network,liste=liste,behavior=behavior)

#create 3D plot to show disconnection index in CC16, and CC10  in patients 
#with or without lVP   
    listes=[C,D,CD]
    x = "CC_16"
    y = "CC_10"
    z= "lVP"
    name = "_16_10_scatter"
    tridim_plot(listes,x,y,z,name)
    
#create 3D plot to show disconnection index in CC16, and CS16 and CT16  
    listes=[C,D,CD]
    x = "CC_16"
    y= "CC_16"
    z= "CT_16"
    name = "_16_16_16"
    tridim_plot(listes,x,y,z,name)
    

#histogram showing distribution of disconnection index for each network 
liste= CD
hue = "lVP"
bins = 0.1
for network in neuro_columns : 
    plot_normalized_histogram(liste,hue,network, bins)
 
    

##############################################################################
#NON PARAMETRIC STATISTICS   
def loop_for_stats (liste_network, liste1, liste2) : 
    result_stats={}
    group_means;
    for network in liste_network :
         result_stats[network]= mannwhitneyu(liste1[network], 
                                             liste2[network])[1]
    result_stats = pd.DataFrame.from_dict(result_stats, orient='index')
    for network in liste_network :
        result_stats.loc[network,liste1.name]= group_means.loc[network,liste1.name]
        result_stats.loc[network,liste2.name]= group_means.loc[network,liste2.name]
    result_stats.to_excel(data_dir + liste1.name + '_'+ liste2.name + 
                          'result_stats.xlsx')

loop_for_stats(neuro_columns, C, D)
##############################################################################
# PREDICT presence ofsevere intraoperative verbal perseverations (siVP) 
# by applying different supervised learning analysis
# select disconnectome data : 


X = EF[neuro_columns]
y = EF["siVP"]
yy = EF["loss in semantic fluency late postop"]
zz = EF["loss in phonemic fluency laste postop"]


# cross-validation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)



### Random forest Classifier
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : 
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
rf = RandomForestClassifier(random_state=0)

# Nested CV with parameter optimization
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv, 
                   refit='Accuracy')
nested_score_rf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_rf.mean(),nested_score_rf.std()))
gs = clf.fit(X,y)
featimp=gs.best_estimator_.feature_importances_
featimp=pd.DataFrame(featimp)
featimp['neuro_columns']=neuro_columns
featimp.to_excel(data_dir + '/featimp.xlsx')
importances = gs.best_estimator_.feature_importances_
importances=pd.DataFrame(importances)
importances['neuro_columns']=neuro_columns
std = np.std([gs.tree_.feature_importances_ for tree in gs.estimators_],
             axis=0)
#featimp = np.argsort(featimp.iloc[:,0])[::-1]
featimp = featimp.rename(columns={0 : 'importances' })
featimp = featimp.sort_values(by=['importances'], ascending = False)
plotfeatimp = plt.figure()
my_dpi = 300
plt.figure(figsize=(10000 / my_dpi, 5000 / my_dpi), dpi=my_dpi)
plt.title("Feature importances", fontsize=30)
plt.bar(range(X.shape[1]), featimp.iloc[:,0],
        color="r" , align="center")
plt.xticks(range(X.shape[1]), featimp.iloc[:,1],fontsize=20)
plt.xlim([-1, X.shape[1]])
plt.yticks(fontsize=20)
plt.show()

### Random forest Regressor to predict semantic fluency late postop

X, yy = make_regression(n_features=26, n_informative=2,
                        random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : 
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}

# Nested CV with parameter optimization
clf = GridSearchCV(estimator=regr, param_grid=p_grid, cv=inner_cv,
                   refit='Score')
nested_score_rf = cross_val_score(clf, X=X, y=yy, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_rf.mean(),nested_score_rf.std()))
gs = clf.fit(X,yy)
featimp=gs.best_estimator_.feature_importances_
featimp=pd.DataFrame(featimp)
featimp['neuro_columns']=neuro_columns
featimp.to_excel(data_dir + '/featimp.xlsx')
std = np.std([gs.tree_.feature_importances_ for tree in gs.estimators_],
             axis=0)
featimp = featimp.rename(columns={0 : 'importances' })
featimp = featimp.sort_values(by=['importances'], ascending = False)
plotfeatimp = plt.figure()
my_dpi = 300
plt.figure(figsize=(10000 / my_dpi, 5000 / my_dpi), dpi=my_dpi)
plt.title("Feature importances", fontsize=30)
plt.bar(range(X.shape[1]), featimp.iloc[:,0],
        color="r" , align="center")
plt.xticks(range(X.shape[1]), featimp.iloc[:,1],fontsize=20)
plt.xlim([-1, X.shape[1]])
plt.yticks(fontsize=20)
plt.show()

### Random forest Regressor to predict phonemic fluency late postop

X, zz = make_regression(n_features=26, n_informative=2,
                        random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : 
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}

# Nested CV with parameter optimization
clf = GridSearchCV(estimator=regr, param_grid=p_grid, cv=inner_cv, 
                   refit='Score')
nested_score_rf = cross_val_score(clf, X=X, y=zz, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_rf.mean(),nested_score_rf.std()))
gs = clf.fit(X,zz)
featimp=gs.best_estimator_.feature_importances_
featimp=pd.DataFrame(featimp)
featimp['neuro_columns']=neuro_columns
featimp.to_excel(data_dir + '/featimp.xlsx')

std = np.std([gs.tree_.feature_importances_ for tree in gs.estimators_],
             axis=0)
featimp = featimp.rename(columns={0 : 'importances' })
featimp = featimp.sort_values(by=['importances'], ascending = False)
plotfeatimp = plt.figure()
my_dpi = 300
plt.figure(figsize=(10000 / my_dpi, 5000 / my_dpi), dpi=my_dpi)
plt.title("Feature importances", fontsize=30)
plt.bar(range(X.shape[1]), featimp.iloc[:,0],
        color="r" , align="center")
plt.xticks(range(X.shape[1]), featimp.iloc[:,1],fontsize=20)
plt.xlim([-1, X.shape[1]])
plt.yticks(fontsize=20)
plt.show()

# TREE
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
clf.score(X_test, y_test)
r = export_text(clf, list(X.columns))
print(r)

# Nested cross-validation
nested_score_rf = np.zeros(4)

# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "min_samples_leaf" : [1, 2, 3, 4, 5]}
rf = tree.DecisionTreeClassifier(random_state=0)

# Nested CV with parameter optimization
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
nested_score_tree = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(
    nested_score_rf.mean(),
    nested_score_rf.std()))

clf.estimator.fit(X, y)
r = export_text(clf.estimator, list(X.columns))
print(r)   
