#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:18:40 2018

@author: ben
"""

import numpy as np
import pandas as pd
#import seaborn as sns


"""Change to R_H (Healthy reference population) and R_D (Disease reference population)
--> p_D := prevalence"""


"""Ref1 := cases; Ref2 := non-cases; Mix := mixture"""


def load_diabetes_data(metric):

    dataset = 'data/biobank_mix_WTCC_ref.csv'
#    metrics = ['T1GRS', 'T2GRS']
    headers = {'diabetes_type': 'group', 't1GRS': 'T1GRS', 't2GRS': 'T2GRS'}

    if metric == 'T1GRS':
        # bin_width = 0.006  # 0.005
        # bin_min = 0.095
        # bin_max = 0.350

        # TODO: Derive this from the scores (this is close to but not equal to the Ref1_median)
        median = 0.23137931525707245  # Type 2 population (0.231468353)

    elif metric == 'T2GRS':
        # bin_width = 0.1
        # bin_min = 4.7
        # bin_max = 8.9

        median = 6.78826

    else:
        raise ValueError(metric)
    binning_method = 'fd'

    data = pd.read_csv(dataset)
    data.rename(columns=headers, inplace=True)
    # print(data.describe())

    scores = {}
    means = {}
    prop_Ref1 = None

    # Arrays of metric scores for each group
    scores = {'Ref1': data.loc[data['group'] == 1, metric].values,
              'Ref2': data.loc[data['group'] == 2, metric].values,
              'Mix': data.loc[data['group'] == 3, metric].values}
    # Try creating a "ground truth" by combining both reference populations
              # 'Mix': np.r_[data.loc[data['group'] == 1, metric].values,
              #              data.loc[data['group'] == 2, metric].values]}

    # if metric == 'T1GRS':
    #     prop_Ref1 = len(scores['Ref1'])/(len(scores['Ref1']) + len(scores['Ref2']))

    means = {'Ref1': scores['Ref1'].mean(),
             'Ref2': scores['Ref2'].mean()}

    medians = {"Ref1": np.median(scores["Ref1"]),
               "Ref2": median}

    # --------------------------- Bin the data -------------------------------
#    N = data.count()[0]
    # bins = {}
    #
    # bin_edges = np.arange(bin_min, bin_max+bin_width, bin_width)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    # bins = {'width': bin_width,
    #         'min': bin_min,
    #         'max': bin_max,
    #         'edges': bin_edges,
    #         'centers': bin_centers}

    # if False:
    #     sns.jointplot(x='T1GRS', y='T2GRS', data=data)
    # #    f, ax = plt.subplots(1, 3)
    #     sns.jointplot(x='T1GRS', y='T2GRS', color='r', data=data.loc[data["group"]==1])
    #     sns.jointplot(x='T1GRS', y='T2GRS', color='b', data=data.loc[data["group"]==2])
    #     sns.jointplot(x='T1GRS', y='T2GRS', color='g', data=data.loc[data["group"]==3])
    # #    sns.JointGrid(x='T1GRS', y='T2GRS', data=data)

    print("Diabetes Data [{}]".format(metric))
    # print("=====================")
    # print("Chosen: width = {}".format(bin_width))
    hist, bins = estimate_bins(scores)
    return scores, bins[binning_method], means, medians, prop_Ref1


def load_renal_data():
    '''Load individuals with Type 2 Diabetes GRS and renal disease (microalbuminuria)
    Group: 1 is controls;
    2 is non insulin treated type 2;
    3 is mixture (microalbuminuria) in all individuals not on insulin.
    Truth is 7.77'''

    metric = 'T2GRS'

    # dataset = 'data/renal_data.csv'
    # headers = {'t2d_grs_77': 'T2GRS', 'group': 'group'}
    dataset = 'data/renal_final_data_mixture_2_type_1_1.xls'
    headers = {"T2GRS": "T2GRS", "Group": "group"}
    prop_Ref1 = 0.0758
    binning_method = 'fd'  # TODO: Take max over Ref1, Ref2?

    # New "enriched" data set
    # dataset = 'data/renal_data_new.csv'
    # headers = {'T2_GRS': 'T2GRS', "Group (1 t2, 2 control, 3 mixture)": 'group'}
    # prop_Ref1 = 0.1053

    # bin_width = 0.128  # 0.1
    # bin_min = 4.7
    # bin_max = 8.9

#    data = pd.read_csv(dataset)
    data = pd.read_excel(dataset)
    data.rename(columns=headers, inplace=True)
    data.dropna(inplace=True)  # Remove empty entries
    # print(data.describe())

    scores = {}
    means = {}

    # Arrays of metric scores for each group
#    scores = {#'Ref1': data.loc[data['group'] == 1, metric].sample(n=100000).values,
##              'Ref1': data.loc[data['group'] == 1, metric].values,
#              'Ref1': data.loc[data['group'] == 2, metric].values,
##              'Ref2': data.loc[data['group'] == 1, metric].sample(n=100000).values,
#              'Ref2': data.loc[data['group'] == 1, metric].sample(n=10000, random_state=42).values,
#              # 'Ref2': data.loc[data['group'] == 2, metric].values,
#              'Mix': data.loc[data['group'] == 3, metric].values}

    prev_data = pd.read_csv('data/renal_data.csv')
    prev_data.rename(columns={'t2d_grs_77': 'T2GRS', 'group': 'group'}, inplace=True)

    scores = {'Ref1': data.loc[data['group'] == 1, metric].values,
              'Ref2': prev_data.loc[prev_data['group'] == 1, metric].sample(n=10000, random_state=42).values,
              'Mix': data.loc[data['group'] == 2, metric].values}

    means = {'Ref1': scores['Ref1'].mean(),
             'Ref2': scores['Ref2'].mean()}

    medians = {"Ref1": np.median(scores['Ref1']),
               "Ref2": np.median(scores['Ref2'])}

    # --------------------------- Bin the data -------------------------------
#    N = data.count()[0]
    # bins = {}
    #
    # bin_edges = np.arange(bin_min, bin_max+bin_width, bin_width)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    # bins = {'width': bin_width,
    #         'min': bin_min,
    #         'max': bin_max,
    #         'edges': bin_edges,
    #         'centers': bin_centers}

    print("Renal Data")
    # print("==========")
    # print("Chosen: width = {}".format(bin_width))
    hist, bins = estimate_bins(scores)
    return scores, bins[binning_method], means, medians, prop_Ref1


def load_coeliac_data():
    '''
    non-cases 1,
    cases 2,
    mixture 3
    '''

    dataset = 'data/new_data_control_1_cases_2_mixture_3.csv'
    prop_Ref1 = 0.139  # Not ground truth - 13.9% report a diagnosis of coeliac so our analysis shows there is no undiagnosed coeliac within a gluten free cohort
    binning_method = 'fd'

    data = pd.read_csv(dataset, header=None, names=['GRS', 'group'])
    data.dropna(inplace=True)  # Remove empty entries

    scores = {'Ref1': data.loc[data['group'] == 2, 'GRS'].values,
              'Ref2': data.loc[data['group'] == 1, 'GRS'].values,
              'Mix': data.loc[data['group'] == 3, 'GRS'].values}

    means = {'Ref1': scores['Ref1'].mean(),
             'Ref2': scores['Ref2'].mean()}

    medians = {"Ref1": np.median(scores['Ref1']),
               "Ref2": np.median(scores['Ref2'])}

    print("Coeliac Data")
    # print("==========")
    # print("Chosen: width = {}".format(bin_width))
    hist, bins = estimate_bins(scores)
    return scores, bins[binning_method], means, medians, prop_Ref1


# Let's use FD!
def estimate_bins(data, bin_range=None, verbose=0):
    # 'scott': n**(-1./(d+4))
    # kdeplot also uses 'silverman' as used by scipy.stats.gaussian_kde
    # (n * (d + 2) / 4.)**(-1. / (d + 4))
    # with n the number of data points and d the number of dimensions
    line_width = 49

    hist = {}
    bins = {}
    if verbose:
        print("  Method | Data |  n  |  width  |      range     ", flush=True)
        print("="*line_width)
    for method in ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']:
        all_scores = []
        all_refs = []
        for group, scores in data.items():
            all_scores.extend(scores)
            if group != "Mix":
                all_refs.extend(scores)
            # else:  # Add extremes to ensure the full range is spanned
            #     all_refs.extend([min(scores), max(scores)])
            if bin_range is None:
                bin_range = (min(all_scores), max(all_scores))
            if verbose > 1:
                _, bin_edges = np.histogram(scores, bins=method, range=bin_range)
                print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]".format(method, group, len(bin_edges)-1, bin_edges[1]-bin_edges[0], bin_edges[0], bin_edges[-1]))
                # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format(group, method, bin_edges[1]-bin_edges[0], len(bin_edges)-1, bin_edges[0], bin_edges[-1]))

        h_r, edges_r = np.histogram(all_refs, bins=method,
                                    range=(min(all_scores), max(all_scores)))
        if verbose > 1:
            print("-"*line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "Refs", len(edges_r)-1, edges_r[1]-edges_r[0], edges_r[0], edges_r[-1]))

        h_a, edges_a = np.histogram(all_scores, bins=method, range=bin_range)  # Return edges

        if verbose:
            if verbose > 1:
                print("-"*line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "All", len(edges_a)-1, edges_a[1]-edges_a[0], edges_a[0], edges_a[-1]))
            # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format("All", method, b['width'], b['n'], b['min'], b['max']))
            if verbose > 1:
                print("="*line_width)
            else:
                print("-"*line_width)

        # h, edges = h_a, edges_a
        h, edges = h_r, edges_r
        hist[method] = h
        bins[method] = {'width': edges[1] - edges[0],
                        'min': edges[0],
                        'max': edges[-1],
                        'edges': edges,
                        'centers': (edges[:-1] + edges[1:]) / 2,
                        'n': len(edges) - 1}
    return hist, bins
