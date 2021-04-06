#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn import tree
from sklearn.tree import (DecisionTreeClassifier,export_text,plot_tree)
from sklearn.datasets import make_regression
from sklearn import decomposition
import graphviz 


# Create a directory containing the csv files containing disconnection data,
# imaging files (cavities), as well as clinical data

# replace with wherever you put that folder
if 1:
    data_dir = '/Volumes/MF2/analyse_PV/analyse_PV_2021'
else:
    data_dir = os.getcwd()

# setting it only once to better adapt to local configuration 
csv_file = os.path.join(data_dir, 'all_connections.csv')
csv_file_supp = os.path.join(data_dir, 'suppl_connections.csv')
csv_file_behavior = os.path.join(data_dir, 'behavior_9_08022021.csv')

# directory where figures are saved
write_dir = data_dir
image_dir = data_dir 
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
ANTS_ratio_clean["postop"]=ANTS_ratio_clean["lVP"]+ANTS_ratio_clean["eVP"]

##############################################################################
# CALCULATE VOLUME OF RESECTION CAVITY (in MNI space) FROM .NII  FILE
 
for i in ANTS_ratio_clean.index : 
    INPUT = image_dir + "/"+ ANTS_ratio_clean.loc[i,"Name_image"]

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
    ANTS_ratio_clean.loc[i,"volume_lesion"]= nonzero_voxel_volume
##############################################################################

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
## DD is to remove one subjet with weird postoperative behavior (eVP- & lVP+)
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
Z= pd.concat([C, DD],axis=0)
Z.name = "Z"


#define group list ( J is removed because is empty)
group_list = [A, B, C, D, DD, E, F, G, H, I, K, L, Z]
group_means={}
for group in group_list:  
    group_means[group.name] = group.mean(skipna=True)[:26]
group_means = pd.DataFrame(group_means)
group_means.to_excel(data_dir + '/group_means.xlsx')


##############################################################################
neuro_columns = [
    'CC_10', 'CC_12', 'CC_13', 'CC_14', 'CC_15', 'CC_16', 'CC_17', 'CC_4',
    'CC_5', 'CC_6', 'CC_7', 'CC_8', 'CC_9', 'CS_10', 'CS_12', 'CS_13',
    'CS_16', 'CS_17', 'CS_3', 'CS_4', 'CS_7', 'CS_8', 'CS_9', 'CT_16',
    'CT_7', 'CT_8']

ordered_networks = [
    "CS_3","CC_4","CS_4","CC_5","CC_6","CC_7","CS_7","CT_7","CC_8",
    "CS_8","CT_8","CC_9","CS_9","CC_10","CS_10","CC_12","CS_12",
    "CC_13","CS_13","CC_14","CC_15","CC_16","CS_16","CT_16","CC_17","CS_17"]

## relevant networks defined arbitrarily after inspection of data
most_disconnected_networks = [
        "CS_4","CC_7","CS_7","CT_7","CC_8",
    "CS_8","CT_8","CC_9","CC_10","CS_10","CC_12","CS_12",
    "CC_13","CS_13","CC_14","CC_15","CC_16","CS_16","CT_16","CC_17","CS_17"]
most_relevant_networks = [
    "CS_3","CC_4","CS_4","CC_5","CC_6","CT_7","CS_9","CC_15","CT_16"]
most_relevant_volume =  [
    "CS_3","CC_4","CS_4","CC_5","CC_6","CT_7","CS_9","CC_15","CT_16",
    "volume_lesion"]
less_relevant_networks = [
    "CC_7","CS_7","CC_8","CS_8","CT_8","CC_9","CC_10","CS_10","CC_12","CS_12",
    "CC_13","CS_13","CC_14","CC_16","CS_16","CC_17","CS_17"]

##############################################################################    
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


# Plotting functions

#create scatter plot for each level ( CC16, CS , CT) displaying
# disconnection index in patients with or without lVP
#WORK TO DO : find a way to display name of network on graph
def scatter_network(networks, liste, behav) :
    '''
    Parameters
    ----------
    networks : networks you want to plot (include as many as you want)
    liste : groups of patients to include in the analysis
    behavior : postoperative behavior you're interested in

    Returns
    -------
    For each network , a scatter plot indicating disconnection index in the
    network according to behavior

    '''
    for network in networks : 
        my_dpi = 300
        fig, ax = plt.subplots(figsize=(3000 / my_dpi, 7000/ my_dpi), 
                               dpi=my_dpi, 
                               constrained_layout=True)
        ax.scatter(liste[behav],liste[network], s=5)
        for x,y,z in zip(liste[behav],liste[network],liste.index):
            label = f"({x},{y})"
            plt.annotate(z.split('_')[1],(x,y),textcoords="offset points",
                         xytext=(10,10), size=5,
                         ha='center',bbox=dict(boxstyle = 'round,pad=0.1', 
                                               fc = 'yellow',alpha = 0.5))
        plt.savefig(write_dir+level+"_scatter.png")
        
#create 3D plot to show relation between parameters in different lists

def tridim_plot (liste,network1,network2,network3,behav) : 
    '''
    

    Parameters
    ----------
    liste : group of patients to include in the analysis
    network1 : network you want to plot on the x axis
    network2 : network you want to plot on the y axis
    network3 : network you want to plot on the z axis
    behavior : behavior you want to plot as color 

    Returns
    -------
    For each list, outputs a 3D graph plotting disconnection index in network
    1 (x-axis),
    network2 (y-axis), and network3 (z-axis)
    
 

    '''
    centers = [[1, 1], [-1, -1], [1, -1]]
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    for name, label in [('no_' + behav, 0), (behav, 1)]:
        ax.text3D(liste[network1] [liste[behav] == label].astype(float).mean(),
                  liste[network2] [liste[behav] == label].astype(float).mean()
                  + 1.5,
                  liste[network3] [liste[behav] == label].astype(float).mean()
                  , name,
                  
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    
    y = np.choose(liste[behav].astype(int), [1, 0]).astype(float)
    ax.scatter(liste[network1].astype(float), liste[network2].astype(float),
               liste[network3].astype(float), c=y, cmap="spring",
               edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(network1)
    ax.set_ylabel(network2)
    ax.set_zlabel(network3)
    
    
#plots showing disconnection in one network, lesion volume and postoperative 
behavior
def relplot (liste,network,behav):
    '''
    

    Parameters
    ----------
    liste : group of patients to include in the analysis
    network : network you are interested in
    behavior : postoperative behavior you are interested in

    Returns
    -------
    a scatter plot of disconnection index in the network, 
    behavior is shown as color of points, volume of cavity is shown as size of
    point

    '''
    ax= sns.relplot(1,liste[network], hue = liste[behav], 
                    size = liste["volume_lesion"]/1000)
    plt.savefig(write_dir+ network +"_scatter.png")


def plot_smoothed_distibution(liste,hue,network):
    '''
    

    Parameters
    ----------
    liste : group of patients you are interested in
    hue : what you want to be displayed as hue (usually behavior)
    network : network you are interested in
        

    Returns
    -------
    A Kernel Density Estimate of the distribution of disconnection index in 
    this network

    '''
    sns.displot( liste,x=pd.to_numeric(liste[network]), hue=hue, kind="kde",
                multiple="stack")
    plt.savefig(write_dir + '_' + network + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)

def plot_normalized_histogram(liste,hue,network,bins):
    '''
    

    Parameters
    ----------
    liste : group of patients you are interested in
    hue : what you want to be displayed as hue (usually behavior)
    network : network you are interested in
    bins : size of bins

    Returns
    -------
    A histogram showing the distribution of disconnection index 
    for the given network

    '''
    my_dpi = 300
    fig = plt.figure(figsize=(20000/ my_dpi, 20000 / my_dpi), dpi=my_dpi)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Distribution of disconnection index for each network ")
    sns.displot( liste,x=pd.to_numeric(liste[network]), hue=hue)
    plt.savefig(write_dir + '_' + network + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)


def heatmap (liste,networks,name):
    '''
    

    Parameters
    ----------
     

    liste : list of patients you are interested in
    j : index of the column for the first network we want to plot
    k : index above the last network in the dataframe ( if we want to plot 
    network with index 25,k=26)
    networks : networks you want to plot (include as many as you want, 
                                          or provide a list)
    name = name to be displayed on plot ("Name_image" or "Name_anonym" or
                                         "Nom")

    Returns
    -------
    A heatmap representing disconnection index as a color
    y axis : patients
    x axis : networks 
    

    '''
    data = liste
    data.dropna(subset=["lVP"],inplace=True)
    data= data.sort_values("lVP")
    lVP= data.pop("lVP").astype(str)
    lVP=lVP.str.replace("1.0","late VP")
    lVP=lVP.str.replace("0.0","no late VP")
    if name == "nom" :
        legend = lVP + data.index
    else :
        names= data.pop(name)
        legend=lVP+names
    data= data[networks].astype(float)
    sns.set(font_scale=0.7)
    ax=sns.heatmap(data,cmap="Spectral",cbar=True, vmax=.5, center=0,
                   square=True, linewidths=0.5, 
                   cbar_kws={"orientation": "vertical"},xticklabels=True, 
                   yticklabels=legend)



def pairplot(liste,networks,behav):   
    '''
    

    Parameters
    ----------
    liste : list of subjects you are interested in
    networks : networks you want to plot (include as many as you want,
                                          or provide a list)

    Returns
    -------
    A n by n matrix of scatterplots representing disconnection index 
    of two networks. Different colors are used for subject with lvp and 
    patients without lvp
    The diagonal of the matrix is an histogram representing the distribution 
    of disconnection index for patients with lvp and patients without lvp


    '''

    data= pd.concat([liste[networks], liste[behav]],axis=1)
    g = sns.PairGrid(data, hue=behav )
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()

#histogram showing distribution of disconnection index for each network 

def hist_distrib (liste,networks,behav) : 
    '''
    

    Parameters
    ----------
    liste : list of subjects you are interested in
    networks : list of networks you are interested in
    behav : behavior you are interested in

    Returns
    -------
    A figure with multiple suplots. Each supblot is an histogram representing
    disconnection index in each network. There are two bars (of different 
    color)side to side corresponding to the presence or absence of behavior

    '''
    fig, axes = plt.subplots(7, 4)
    axes=axes.flatten()
    fig.suptitle('distribution of disconnection index for each network')
    i=0
    for network in networks:
       plt.sca(axes[i])
       plt.style.use('seaborn-deep')
       plt.hist([liste[network][liste[behav]==0].astype(float),
                 liste[network][liste[behav]==1].astype(float)], 
                label=[behav+'-',behav+'+'])
       plt.title(network,loc='center', y=0.5)
       i=i+1

def make_spider(networks, row, title, data, color,ax):
    '''
    

    Parameters
    ----------
    categories : networks you want to plot
    row : number of the row corresponding to the subject you are interested in
    (usually defined by the loop_to_plot function)
    title : title you want to be displayed on the spider plot  
    (usually defined by the loop_to_plot
                                          function)
    liste : group of patient you are interested in (usually defined
    by the loop_to_plot function)
    color : color to be used in the plot (usually defined by the loop_to_plot
                                          function)
    ax : ax for plotting (usually defined by the loop_to_plot function)

    Returns
    -------
    A spider plot showing disconnection index for one subject

    '''
    
    n_networks = len(networks)
    # define the angle for each variable
    angles = [n / float(n_networks) * 2 * pi for n in range(n_networks)]
    #angles += angles[:1]   
    # initialize the spider plot
#    ax2 = spider.add_axes( polar=True )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, [i for i in networks], color='black', size=6)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9],
               ["0.1", "0.3", "0.5", "0.7", "0.9"], color="grey", size=7)
    plt.ylim(0, 1)
     
    # Ind1
    values = data.values[row].flatten().tolist()
    #values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2,linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)



    # Add a title
    plt.title(title, size=8, color='grey', y=1.3)

#    plt.subplots_adjust(left=0.125,
#                        bottom=0.1, 
#                        right=0.9, 
#                        top=0.9, 
#                        wspace=0.1, 
#                        hspace=0.4)


def loop_to_plot1(networks, liste1,liste2, m,n):
    '''
    

    Parameters
    ----------
    categories : networks you want to plot
    row : number of the row corresponding to the subject you are interested in
    liste1 : first group of patient you are interested in 
    list 2 : second group of patient you are interested in

    m, n = size of grid for subplotting


    Returns
    -------
   Disconnection index of all subjects are overlaid on the same radar plot : 
       cool colors for subjects without the behavior of interest
       warm colors for subject with the behavior of interest
        
       

    '''
    # initialize the figure
    my_dpi = 300
    fig = plt.figure(figsize=(9000 / my_dpi, 9000 / my_dpi), dpi=my_dpi)
    # Create a color palette 
    my_palette1 = plt.cm.get_cmap("twilight", len(liste1.index))

    data=liste1[networks]
    data=pd.DataFrame(data.mean(axis=0))
    data=data.T
    data= pd.concat([liste2[networks],data], axis=0)

    outer = gridspec.GridSpec(m, n, wspace=0.2, hspace=0.3)
    # Loop to plot
    for row in range(0,len(data.index)):
        ax = fig.add_subplot(outer[row],polar=True)
        make_spider(networks=networks, row=row, data = data,
                     title=data.index[row], 
                     color=my_palette1(row), ax=ax)

    # save figure 
    plt.savefig(write_dir + str(networks[0].partition('_')[0]) + '_' 
            +"moyenne"+"_"+ liste1.name + "_vs_" + liste2.name + '.png',
            dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)


def loop_to_plot2(networks, liste,m,n):
    '''
    

    Parameters
    ----------
    networks: networks you want to plot
    liste : group of patient you are interested in 
    m : first number of gridspec
    n : second number of gridspec

    Returns
    -------
    A figure, with one subplot per subject  which includes : 
        a radar plot showing disconnection index
        the resection cavity in the axial, coronal and sagittal plane

    '''
    # initialize the figure
    my_dpi = 300
    fig = plt.figure(figsize=(9000 / my_dpi, 9000 / my_dpi), dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(liste.index))

   #the numbers in gridspec
    outer = gridspec.GridSpec(m, n, wspace=0.2, hspace=0.3)

    # Loop to plot
    for row in range(0,len(liste.index)):
        inner = gridspec.GridSpecFromSubplotSpec(2,1,
                    subplot_spec=outer[row], wspace=0.1, hspace=0.1)
        ax = fig.add_subplot(inner[0],polar=True)
        make_spider(networks, row=row,data=liste[networks],
                     title=liste.Name_anonym[row], 
                     color=my_palette(row), ax=ax)
        ax2 = fig.add_subplot(inner[1])
        plotting.plot_roi(image_dir + "/" + liste["Name_image"][row],axes=ax2)
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
    plt.savefig(write_dir + str(networks[0].partition('_')[0]) + '_' 
            + liste.name + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, metadata=None)
    

##############################################################################
#NON PARAMETRIC STATISTICS   
def loop_for_stats (networks, liste1, liste2) : 
    '''
    

    Parameters
    ----------
    networks : networks you are interested in (list)
    liste1 :list of subjects you are interested in
    liste2 : list of subjects you are interested in
    Returns
    -------
    an .xlsx file displaying for each network the mean disconnection index
    in each group, as well as the p value of mannwhitney test

    '''
    result_stats={}
    group_means;
    for network in networks :
         result_stats[network]= mannwhitneyu(liste1[network], 
                                             liste2[network])[1]
    result_stats = pd.DataFrame.from_dict(result_stats, orient='index')
    for network in networks :
        result_stats.loc[network,liste1.name]= group_means.loc[network,
                                                               liste1.name]
        result_stats.loc[network,liste2.name]= group_means.loc[network,
                                                               liste2.name]
    result_stats.to_excel(data_dir + liste1.name + '_'+ liste2.name + 
                          'result_stats.xlsx')

##############################################################################
# use machine learning techniques to identify patterns in the dataset


##### Permorm a PCA with 3 components and show results in a tridimensional plot
#initiate figure
def PCA_3D (liste,networks,behav):
    '''
    

    Parameters
    ----------
    liste : group of subjects you are interested in
    networks : list of networks to be included in the analysis
    behavior : behavior you are interested to plot on the 3D graph (as hue)

    Returns
    -------
    Performs a 3D PCA. Returns :
        -explained variance for each component in the console
        -a 3D scatter plot representing each datapoint along the 3axis of the
        PCA. Datapoints are shown in different colors according to behavior
        -an histogram showing the importance of each network in the first 
        component

    '''
    #select data
    X = liste[networks].astype(float)
    y = liste[behav]
    y = np.asarray(y).astype(int)    
    
    #initiate figure
    centers = [[1, 1], [-1, -1], [1, -1]]
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    #perform PCA
    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    X = pca.transform(X)
    
    #3D scatter plot accord to PCA axis (n=3)
    for name, label in [('no_' + behav, 0), (behav, 1)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(),name,
                  
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1],X[:, 2], c=y, cmap="spring",
               edgecolor='k')
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    
    plt.show()
    #show the networks with the most importance in the first component
    components=pd.DataFrame(pca.components_)
    components.columns = most_disconnected_networks
    components=components.iloc[0,:].T
    components = components.sort_values( ascending = False)
    
    
    plotcomponents = plt.figure()
    my_dpi = 300
    plt.figure(figsize=(10000 / my_dpi, 5000 / my_dpi), dpi=my_dpi)
    plt.title("networks_1stcomponent", fontsize=30)
    plt.bar(components.index, components,color="r" , align="center")
    plt.xticks(range(components), len(components),fontsize=20)
    plt.xlim([-1, range(components)])
    plt.yticks(fontsize=20)
    plt.show()

##### perform a PCA on a 2-dimensional dataset and plot the results
def PCA_2D (liste,network1,network2) :
    '''
    

    Parameters
    ----------
    liste : group of subject you are interested in 
    network1 : first network you're interested in (string)
    network2 : second you're interested in (string).

    Returns
    -------
    Performs a two-dimensional PCA 
    Returns : 
        -explained variance for each component in the console
        -a 2 dimension scatterplot of the data, with the first and second 
        component of thePCA plotted on the scatter plot
    '''
    #select data
    X = liste[[network1,network2]].astype(float)
    #pca
    pca = PCA(n_components=2).fit(X)
    #printresults
    print(pca.explained_variance_ratio_)
    #plot
    plt.scatter(X.iloc[:,1 ], X.iloc[:, 1], alpha=.3, label='samples')
    for i, (comp, var) in enumerate(zip(pca.components_, 
                                        pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", 
                 linewidth=5,
                 color=f"C{i + 2}")
    plt.gca().set(aspect='equal',
                  title="2-dimensional dataset with principal components",
                  xlabel='first feature', ylabel='second feature')
    plt.legend()
    plt.show()


### the code works but no when executed as a function
def arbre (liste, networks, behav) :
    '''
    

    Parameters
    ----------
    liste : group of subjects you are interested in
    networks : list of networks to be included in the analysis
    behavior : behavior you are interested to plot on the 3D graph (string)

    Returns
    -------
    computes a decision tree based on the integrality of data ( no validation 
                                                               approach)
    returns : 
        -tree as a textoutput
        -tree as a graph

    '''
    
    X = liste[networks].astype(float)
    y = liste[behav]
    y = np.asarray(y).astype(int)   


    # TREE
    
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    
    r = export_text(clf, list(X.columns))
    print(r)

    dot_data = tree.export_graphviz(clf, feature_names=X.columns,
                                    class_names=[behav + '-', behav + "+"],
                                    filled=True, label="root", rounded=True, 
                                    special_characters=True, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph





def rf(liste,networks,behav) : 
    '''
    

    Parameters
    ----------
    liste : group of subjects you are interested in
    networks : list of networks to be included in the analysis
    behavior : behavior you are interested to plot on the 3D graph (string)
    Returns
    -------
    Performs a Random Forrest analysis with crossvalidation
    Returns mean score of classifier in console
    Outputs an histogram displaying feature importances

    '''
    ### Random forest Classifier
    X = liste[networks].astype(float)
    y = liste[behav]
    y = np.asarray(y).astype(int)  
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
    
    # Set up possible values of parameters to optimize over
    p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : 
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
    rf = RandomForestClassifier(random_state=0)
    
    # Nested CV with parameter optimization
    clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv, 
                       refit='Accuracy')
    nested_score_rf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(
        nested_score_rf.mean(),nested_score_rf.std()))
    
    gs = clf.fit(X,y)
    featimp=gs.best_estimator_.feature_importances_
    featimp=pd.DataFrame(featimp)
    featimp['networks']=networks
    featimp.to_excel(data_dir + '/featimp.xlsx')
    #std = np.std([gs.tree_.feature_importances_ for tree in gs.estimators_]
    #,axis=0)
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


