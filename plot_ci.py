#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:40:06 2018

@author: ben
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

@author: ben
"""

import os
import time
#import math
#import sys
#import multiprocessing
from collections import defaultdict, OrderedDict


import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import lmfit

from joblib import Parallel, delayed, cpu_count
# from joblib import Memory
# mem = Memory(cachedir='/tmp')
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import analyse_mixture as pe

import warnings



def plot_distributions(scores, bins, data_label): #, kernel, bandwidth):
#    f, ax = plt.subplots(3, 1)
    plt.figure()
    sns.distplot(scores['Ref1'], bins=bins['edges'], norm_hist=False, label="Ref1: N={}".format(len(scores['Ref1'])))
    sns.distplot(scores['Ref2'], bins=bins['edges'], norm_hist=False, label="Ref2: N={}".format(len(scores['Ref2'])))
    sns.distplot(scores['Mix'], bins=bins['edges'], norm_hist=False, label="Mix: N={}".format(len(scores['Mix'])))
#    sns.despine(top=True, bottom=True, trim=True)
#    if len(indep_vars) > 1:
        # Use jointplot
    plt.legend()
    plt.savefig('figs/distributions_{}.png'.format(data_label))


def load_diabetes_data(metric):

    dataset = 'data/biobank_mix_WTCC_ref.csv'
#    metrics = ['T1GRS', 'T2GRS']
    headers = {'diabetes_type': 'group', 't1GRS': 'T1GRS', 't2GRS': 'T2GRS'}

    if metric == 'T1GRS':
        bin_width = 0.005
        bin_min = 0.095
        bin_max = 0.350

        # NOTE: This is close to but not equal to the Ref1_median
        median = 0.23137931525707245  # Type 2 population

    elif metric == 'T2GRS':
        bin_width = 0.1
        bin_min = 4.7
        bin_max = 8.9

        median = 6.78826

    else:
        raise ValueError(metric)

    data = pd.read_csv(dataset)
    data.rename(columns=headers, inplace=True)
    data.describe()

    scores = {}
    means = {}

    # Arrays of metric scores for each group
    scores = {'Ref1': data.loc[data['group'] == 1, metric].values,
              'Ref2': data.loc[data['group'] == 2, metric].values,
              'Mix': data.loc[data['group'] == 3, metric].values}

    means = {'Ref1': scores['Ref1'].mean(),
             'Ref2': scores['Ref2'].mean()}

    medians = {"Ref1": np.median(scores["Ref1"]),
               "Ref2": median}

    # --------------------------- Bin the data -------------------------------
#    N = data.count()[0]
    bins = {}

    bin_edges = np.arange(bin_min, bin_max+bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = {'width': bin_width,
            'min': bin_min,
            'max': bin_max,
            'edges': bin_edges,
            'centers': bin_centers}


    sns.jointplot(x='T1GRS', y='T2GRS', data=data)
#    f, ax = plt.subplots(1, 3)
    sns.jointplot(x='T1GRS', y='T2GRS', color='r', data=data.loc[data["group"]==1])
    sns.jointplot(x='T1GRS', y='T2GRS', color='b', data=data.loc[data["group"]==2])
    sns.jointplot(x='T1GRS', y='T2GRS', color='g', data=data.loc[data["group"]==3])
#    sns.JointGrid(x='T1GRS', y='T2GRS', data=data)

    prop_Ref1 = None
    return scores, bins, means, medians, prop_Ref1


def load_renal_data():

    '''Load individuals with Type 2 Diabetes GRS and renal disease (microalbuminuria)
    Group: 1 is controls; 2 is non insulin treated type 2; 3 is mixture (microalbuminuria) in all individuals not on insulin.
    Truth is 7.77'''

    dataset = 'data/renal_data.csv'
    metric = 'T2GRS'
    prop_Ref1 = 0.0777
    headers = {'t2d_grs_77': 'T2GRS', 'group': 'group'}

    bin_width = 0.1  # 0.1
    bin_min = 4.7
    bin_max = 8.9

    data = pd.read_csv(dataset)
    data.rename(columns=headers, inplace=True)
    data.dropna(inplace=True)  # Remove empty entries
    data.describe()

    scores = {}
    means = {}

    # Arrays of metric scores for each group
    scores = {#'Ref1': data.loc[data['group'] == 1, metric].sample(n=100000).values,
#              'Ref1': data.loc[data['group'] == 1, metric].values,
              'Ref1': data.loc[data['group'] == 2, metric].values,
              'Ref2': data.loc[data['group'] == 1, metric].sample(n=100000).values,
              'Mix': data.loc[data['group'] == 3, metric].values}

    means = {'Ref1': scores['Ref1'].mean(),
             'Ref2': scores['Ref2'].mean()}

    medians = {"Ref1": np.median(scores['Ref1']),
               "Ref2": np.median(scores['Ref2'])}

    # --------------------------- Bin the data -------------------------------
#    N = data.count()[0]
    bins = {}

    bin_edges = np.arange(bin_min, bin_max+bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bins = {'width': bin_width,
            'min': bin_min,
            'max': bin_max,
            'edges': bin_edges,
            'centers': bin_centers}

    return scores, bins, means, medians, prop_Ref1





# NOTE: KDEs are very expensive when large arrays are passed to score_samples
# Increasing the tolerance: atol and rtol speeds the process up significantly

#if __name__ == "__main__":

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The 'normed' kwarg is deprecated")

#    mpl.style.use('seaborn')
mpl.rc('figure', figsize=(10, 8))
mpl.rc('font', size=14)
mpl.rc('axes', titlesize=14)     # fontsize of the axes title
mpl.rc('axes', labelsize=14)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=12)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=12)    # fontsize of the tick labels
mpl.rc('legend', fontsize=11)    # legend fontsize
mpl.rc('figure', titlesize=14)  # fontsize of the figure title


# ---------------------------- Define constants ------------------------------

verbose = False
plot_results = False
out_dir = "results_test"

# TODO: Reimplement this
adjust_excess = True

seed = 4242
bootstraps = 1000#1000
alpha = 0.05

KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# ----------------------------------------------------------------------------

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Set random seed
np.random.seed(seed)
# rng = np.random.RandomState(42)

np.seterr(divide='ignore', invalid='ignore')

if True:
    #metric = 'T1GRS'
    data_label ='Diabetes'
    (scores, bins, means, medians, prop_Ref1) = load_diabetes_data('T1GRS')
else:
    data_label ='Renal'
    (scores, bins, means, medians, prop_Ref1) = load_renal_data()
plot_distributions(scores, bins, data_label)  #KDE_kernel, bins['width'])

#for (scores, bins, means, median) in ...
pe.plot_kernels(scores, bins)

if adjust_excess:
    adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
else:
    adjustment_factor = 1.0

methods = OrderedDict([("Means", {'Ref1': means['Ref1'],
                                  'Ref2': means['Ref2']}),
                       ("Excess", {"Median_Ref1": medians["Ref1"],
                                   "Median_Ref2": medians["Ref2"],
                                   "adjustment_factor": adjustment_factor}),
                       ("EMD", True),
                       ("KDE", {'kernel': KDE_kernel,
                                'bandwidth': bins['width']})])
#del methods["KDE"]
#if "Excess" in methods:
#    if isinstance(methods["Excess"], float):
#        # Median passed
#        median = methods["Excess"]
#    else:
#        median = np.median(scores)

#run_method = OrderedDict([("Means", True),
#                          ("Excess", True),
#                          ("EMD", True),
#                          ("KDE", True)])

print("Running mixture analysis with {} scores...".format(data_label))
t = time.time()  # Start timer


(res, df_bs) = pe.analyse_mixture(scores, bins, methods,
                                  bootstraps=bootstraps, alpha=alpha,
                                  true_prop_Ref1=prop_Ref1) #,
#                                  means=None, median=median, KDE_kernel=KDE_kernel)

elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

# Save results
np.save('{}/results_{}'.format(out_dir, data_label), res)
if df_bs is not None:
    df_bs.to_pickle('{}/bootstraps_{}'.format(out_dir, data_label))

# Plot swarm box
f, ax = plt.subplots()
ax = sns.boxplot(data=df_bs)
ax = sns.swarmplot(data=df_bs, color=".25")
sns.despine(top=True, bottom=True, trim=True)
plt.savefig('figs/boxes_{}.png'.format(data_label))

f, ax = plt.subplots()
sns.violinplot(data=df_bs) #, inner="points")
sns.despine(top=True, bottom=True, trim=True)
#plt.hlines()
plt.savefig('figs/violins_{}.png'.format(data_label))
