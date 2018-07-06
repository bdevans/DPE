#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

@author: ben
"""

import os
import time
#import math
#import sys
from collections import defaultdict, OrderedDict

from pprint import pprint

import numpy as np
#import scipy as sp
import pandas as pd
from sklearn.neighbors import KernelDensity
# TODO: replace with a scipy/numpy function to reduce dependencies
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
import lmfit
# TODO: Try replacing with scipy.optimize.curve_fit to solve jopblib problem and reduce dependencies:
# https://lmfit.github.io/lmfit-py/model.html
from joblib import Parallel, delayed, cpu_count
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def fit_kernels(scores, bw):
    kernels = {}
    for label, data in scores.items():
        kernels[label] = {}
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            kde = KernelDensity(kernel=kernel, bandwidth=bw, atol=0, rtol=1e-4).fit(X)  #
            kernels[label][kernel] = kde
    return kernels


#def plot_kernels_(scores, bins):
#    kernels = ['gaussian', 'tophat', 'epanechnikov',
#               'exponential', 'linear', 'cosine']
#    fig, axes = plt.subplots(len(scores), 1, sharex=True) #, squeeze=False)
#    X_plot = bins['centers'][:, np.newaxis]
#
#    kdes = {}
#    for (label, data), ax in zip(scores.items(), axes):
#        kdes[label] = {}
#        X = data[:, np.newaxis]
#        for kernel in kernels:
#            kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(X)
#            log_dens = kde.score_samples(X_plot)
#            ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
#                    label="kernel = '{0}'; bandwidth = {1}".format(kernel, bins['width']))
#            kdes[label][kernel] = kde  # np.exp(log_dens)
#        ax.legend(loc='upper left')
#        ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
#        ax.set_ylabel(label)


def plot_kernels(scores, bins):

    kernels = fit_kernels(scores, bins['width'])
    fig, axes = plt.subplots(len(scores), 1, sharex=True)
    X_plot = bins['centers'][:, np.newaxis]
    for (label, data), ax in zip(scores.items(), axes):
        X = data[:, np.newaxis]
        for name, kernel in kernels[label].items():
            ax.plot(X_plot[:, 0], np.exp(kernel.score_samples(X_plot)), '-',
                    label="kernel = '{0}'; bandwidth = {1}".format(name, bins['width']))
        ax.legend(loc='upper left')
        ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
        ax.set_ylabel(label)


# @mem.cache
def fit_KDE_model(Mix, bins, model, params_mix, kernel):
# TODO: Think carefully about this!
    # x_KDE = np.linspace(bins['min'], bins['max'], len(Mix)+2)
    x_KDE = bins["centers"]
    mix_kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(Mix[:, np.newaxis])
    res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
    amp_Ref1 = res_mix.params['amp_Ref1'].value
    amp_Ref2 = res_mix.params['amp_Ref2'].value
    return amp_Ref1/(amp_Ref1+amp_Ref2)


def interpolate_CDF(scores, x_i, min_edge, max_edge):
    # TODO: x = [x_i[0], *sorted(scores), x_i[-1]]
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def analyse_mixture(scores, bins, methods, bootstrap=1000, true_prop_Ref1=None, means=None, median=None, KDE_kernel='gaussian'):

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']
    bin_width = bins['width']
    bin_edges = bins['edges']

#    print('Running {} mixture analysis...'.format(tag))
#    print('--------------------------------------------------------------------------------')

#    if run_method["Excess"]:
#        # Excess method median of GRS from the whole population in Biobank
#        high = len(Mix[Mix > population_median]) - len(Mix[Mix <= population_median])
#        low = 2*len(Mix[Mix <= population_median])

    # -------------------------------- EMD method --------------------------------

#    if run_method["EMD"]:
    if "EMD" in methods:

        max_EMD = bin_edges[-1] - bin_edges[0]

        # Interpolate the cdfs at the same points for comparison
        i_CDF_Ref1 = interpolate_CDF(Ref1, bins['centers'], bins['min'], bins['max'])
        i_CDF_Ref2 = interpolate_CDF(Ref2, bins['centers'], bins['min'], bins['max'])
#        i_CDF_Mix = interpolate_CDF(Mix, bin_centers, bins['min'], bins['max'])

        # EMDs computed with interpolated CDFs
        i_EMD_1_2 = sum(abs(i_CDF_Ref1-i_CDF_Ref2))
#        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
#        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
#        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD


    if "KDE" in methods:
        # ------------------------------ KDE method ------------------------------

        bw = bin_width  # Bandwidth
        kdes = fit_kernels(scores, bw)

        try:
            KDE_kernel = methods["KDE"]["kernel"]
        except (KeyError, TypeError):
            print("No kernel specified!")
            KDE_kernel = "gaussian"  # Default kernel
        else:
            try:
                bw = methods["KDE"]["bandwidth"]
            except (KeyError, TypeError):
                bw = bins["width"]
        finally:
            print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))

        # Define the KDE models
        # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
        def kde_Ref1(x, amp_Ref1):
            return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

        def kde_Ref2(x, amp_Ref2):
            return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))

        model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

        params_mix = model.make_params()
        params_mix['amp_Ref1'].value = 1
        params_mix['amp_Ref1'].min = 0
        params_mix['amp_Ref2'].value = 1
        params_mix['amp_Ref2'].min = 0


    extra_args = {}

    if "KDE" in methods:
        extra_args['model'] = model  # This breaks joblib
        extra_args['params_mix'] = params_mix
        extra_args['KDE_kernel'] = KDE_kernel
        extra_args['bin_width'] = bin_width
        extra_args['kdes'] = kdes
#        extra_args["fit_KDE"] = fit_KDE

    if "EMD" in methods:
        extra_args['max_EMD'] = max_EMD
        extra_args['i_CDF_Ref1'] = i_CDF_Ref1
        extra_args['i_CDF_Ref2'] = i_CDF_Ref2
#        extra_args['i_EMD_21'] = i_EMD_21
        extra_args['i_EMD_1_2'] = i_EMD_1_2

    if "Means" in methods:
        try:
            Mean_Ref1 = methods["Means"]["Ref1"]
        except (KeyError, TypeError):
            print("No Mean_Ref1 specified!")
            Mean_Ref1 = Ref1.mean()
        finally:
            extra_args["Ref1_mean"] = Mean_Ref1
        try:
            Mean_Ref2 = methods["Means"]["Ref2"]
        except (KeyError, TypeError):
            print("No Mean_Ref2 specified!")
            Mean_Ref2 = Ref2.mean()
        finally:
            extra_args["Ref2_mean"] = Mean_Ref2

    if "Excess" in methods:
        # TODO: Check and rename to Ref1_median?
        # NOTE: This is close to but not equal to the Ref1_median
        if isinstance(methods["Excess"], float):
            # Median passed
            median = methods["Excess"]
            print("Passed median: {}".format(median))
        else:
            median = np.median(scores["Ref1"])
        extra_args['population_median'] = median
        print("Ref1 median: {}".format(median))  # np.median(Ref1)))

    sample_size = len(Mix)
    extra_args['bins'] = bins
#    print(extra_args)


    def estimate_Ref1(RM, Ref1, Ref2, methods, **kwargs):
        '''Estimate the proportion of two reference populations in an unknown mixture.
        The returned proportions are with respect to Ref 1. The proportion of Ref 2 is assumed to be 1 pr(Ref1). '''


        bins = kwargs['bins']
        results = {}

        # ---------------------- Difference of Means method ----------------------
            proportion_of_Ref1 = (RM.mean()-kwargs['Ref2_mean'])/(kwargs['Ref1_mean']-kwargs['Ref2_mean'])
        if "Means" in methods:
            results['Means'] = abs(proportion_of_Ref1)

        # -------------------------- Subtraction method --------------------------
        if "Excess" in methods:
            # TODO: Flip these around for when using the T2GRS
            number_low = len(RM[RM <= kwargs['population_median']])
            number_high = len(RM[RM > kwargs['population_median']])
            sample_size = len(RM)
    #        proportion_Ref1 = (number_high - number_low)/sample_size
            results['Excess'] = (number_high - number_low)/sample_size #kwargs['sample_size']
            results['Excess'] /= 0.92  # adjusted for fact underestimates by 8%


        # ------------------------------ KDE method ------------------------------
        if "KDE" in methods:
            # TODO: Print out warnings if goodness of fit is poor?
            results['KDE'] = fit_KDE_model(RM, bins, model, kwargs['initial_params'], kwargs['KDE_kernel'])

        # ------------------------------ EMD method ------------------------------
        if "EMD" in methods:
            # Interpolated cdf (to compute EMD)
            i_CDF_Mix = interpolate_CDF(RM, bins['centers'], bins['min'], bins['max'])
    #            x = [bins['min'], *np.sort(RM), bins['max']]
    #            y = np.linspace(0, 1, num=len(x), endpoint=True)
    #            (iv, ii) = np.unique(x, return_index=True)
    #            si_CDF_Mix = np.interp(bin_centers, iv, y[ii])

            # Compute EMDs
#            i_EMD_M_1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD #kwargs['max_EMD']
#            i_EMD_M_2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD #kwargs['max_EMD']
#            results["EMD"] = 1 - (i_EMD_M_1 / (i_EMD_M_1 + i_EMD_M_2))

            i_EMD_M_1 = sum(abs(i_CDF_Mix-i_CDF_Ref1))
            i_EMD_M_2 = sum(abs(i_CDF_Mix-i_CDF_Ref2))
            i_EMD_1_2 = sum(abs(i_CDF_Ref1-i_CDF_Ref2))
            results["EMD"] = 0.5 * (1 + (i_EMD_M_2 - i_EMD_M_1)/i_EMD_1_2)
            # print('Proportions based on counts')
            # print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
            # print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))

            # print("Proportions based on Earth Mover's Distance (histogram values):")
            # print('% of Type 1:', 1-EMD_31/EMD_21)
            # print('% of Type 2:', 1-EMD_32/EMD_21)
            #
            # print("Proportions based on Earth Mover's Distance (interpolated values):")
            # print('% of Type 1:', 1-i_EMD_31/i_EMD_21)
            # print('% of Type 2:', 1-i_EMD_32/i_EMD_21)

        return results


    def bootstrap_mixture(sample_size, prop_Ref1, Ref1, Ref2, method, **extra_args):

        assert(0.0 <= prop_Ref1 <= 1.0)
        n_Ref1 = int(round(sample_size * prop_Ref1))
        n_Ref2 = sample_size - n_Ref1

        # Bootstrap mixture
        Mix = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                              np.random.choice(Ref2, n_Ref2, replace=True)))

        run_method = defaultdict(bool)
        run_method[method] = True

        results = estimate_Ref1(Mix, Ref1, Ref2, run_method, **extra_args)

        return results[method]



    # Get initial estimate of proportions
    initial_results = estimate_Ref1(Mix, Ref1, Ref2, methods, **extra_args)
    pprint(initial_results)


    if bootstrap:

        #    nprocs = multiprocessing.cpu_count()
        nprocs = cpu_count()
        print('Running {} bootstraps with {} processors...'.format(bootstrap, nprocs))

        results = OrderedDict()

#        methods = [method for method in ["Means", "Excess", "EMD", "KDE"] if run_method[method]]

        # Spawn threads
#        with Parallel(n_jobs=nprocs) as parallel:
#        with Parallel(n_jobs=1) as parallel:
#            # Parallelise over bootstraps
#
#            # Fix estimated proportions for each method
#            for method in methods:
#                prop_Ref1 = initial_results[method]
#                results[method] = parallel(delayed(bootstrap_mixture)(sample_size, prop_Ref1, Ref1, Ref2, method, **extra_args)
#                                           for b in range(bootstrap))

        for method in methods:
            if true_prop_Ref1:
                prop_Ref1 = true_prop_Ref1
            else:
                prop_Ref1 = initial_results[method]
            results[method] = [bootstrap_mixture(sample_size, prop_Ref1, Ref1, Ref2, method, **extra_args)
                               for b in range(bootstrap)]
        # Put into dataframe
        df_bs = pd.DataFrame(results)


    # ------------ Summarise proportions for the whole distribution --------------
    print()
    print("{:20} | {:^17s} | {:^17s} ".format("Proportion Estimates", "Reference 1", "Reference 2"))
    print("="*61)
    for method in methods:
        print("{:20} | {:<17.5f} | {:<17.5f}".format(method, initial_results[method], 1-initial_results[method]))
        if bootstrap:
            print("{:20} | {:.5f} +/- {:.3f} | {:.5f} +/- {:.3f}".format("Bootstraps (µ±σ)", np.mean(df_bs[method]), np.std(df_bs[method]), 1-np.mean(df_bs[method]), np.std(1-df_bs[method])))  #  (+/- SD)
        print("-"*61)
    if true_prop_Ref1:
        print("{:20} | {:<17.5f} | {:<17.5f}".format("Ground Truth", true_prop_Ref1, 1-true_prop_Ref1))
        print("="*61)
    print()

    if bootstrap:
        return (initial_results, df_bs)
    else:
        return (initial_results, None)
