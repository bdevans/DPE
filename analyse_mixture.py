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
#from statsmodels.nonparametric.kde import KDEUnivariate
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

    extra_args = {}
    sample_size = len(Mix)
    extra_args['bins'] = bins
#    print('Running {} mixture analysis...'.format(tag))
#    print('--------------------------------------------------------------------------------')

    if "Means" in methods:
        try:
            Mean_Ref1 = methods["Means"]["Ref1"]
        except (KeyError, TypeError):
            print("No Mean_Ref1 specified!")
            Mean_Ref1 = Ref1.mean()
        finally:
            extra_args["Mean_Ref1"] = Mean_Ref1
        try:
            Mean_Ref2 = methods["Means"]["Ref2"]
        except (KeyError, TypeError):
            print("No Mean_Ref2 specified!")
            Mean_Ref2 = Ref2.mean()
        finally:
            extra_args["Mean_Ref2"] = Mean_Ref2


    if "Excess" in methods:
    # ----------------------------- Excess method -----------------------------
        # TODO: Check and rename to Ref1_median?

        if isinstance(methods["Excess"], dict):
            if "Median_Ref1" not in methods["Excess"]:
                methods["Excess"]["Median_Ref1"] = np.median(scores["Ref1"])
            if "Median_Ref2" not in methods["Excess"]:
                methods["Excess"]["Median_Ref2"] = np.median(scores["Ref2"])
            median = methods["Excess"]["Median_Ref2"]
            if "adjustment_factor" not in methods["Excess"]:
                methods["Excess"]["adjustment_factor"] = 1
            extra_args['adjustment_factor'] = methods["Excess"]["adjustment_factor"]
#        if isinstance(methods["Excess"], float):
#            # Median passed
#            median = methods["Excess"]
#            print("Passed median: {}".format(median))
#        else:
#            # The Excess method assumes that...
#            median = np.median(scores["Ref2"])
        extra_args['population_median'] = median
        print("Ref1 median:", np.median(Ref1))
        print("Ref2 median:", np.median(Ref2))
        print("Population median: {}".format(median))
        print("Mixture size:", sample_size)


    if "EMD" in methods:
    # -------------------------------- EMD method --------------------------------

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

        extra_args['max_EMD'] = max_EMD
        extra_args['i_CDF_Ref1'] = i_CDF_Ref1
        extra_args['i_CDF_Ref2'] = i_CDF_Ref2
#        extra_args['i_EMD_21'] = i_EMD_21
        extra_args['i_EMD_1_2'] = i_EMD_1_2

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

        extra_args['model'] = model  # This breaks joblib
        extra_args['initial_params'] = params_mix
        extra_args['KDE_kernel'] = KDE_kernel
        extra_args['bin_width'] = bin_width
        extra_args['kdes'] = kdes
#        extra_args["fit_KDE_model"] = fit_KDE_model

#    print(extra_args)


    def estimate_Ref1(RM, Ref1, Ref2, methods, **kwargs):
        '''Estimate the proportion of two reference populations in an unknown mixture.
        The returned proportions are with respect to Ref 1. The proportion of Ref 2 is assumed to be 1 pr(Ref1). '''

        bins = kwargs['bins']
        results = {}

        # ---------------------- Difference of Means method ----------------------
        if "Means" in methods:
            proportion_of_Ref1 = (RM.mean()-kwargs['Mean_Ref2'])/(kwargs['Mean_Ref1']-kwargs['Mean_Ref2'])
            results['Means'] = abs(proportion_of_Ref1)

        # -------------------------- Subtraction method --------------------------
        if "Excess" in methods:
            # TODO: Flip these around for when using the T2GRS
            Median_Mix = np.median(Mix)
            if abs(methods["Excess"]["Median_Ref2"] - Median_Mix) < abs(methods["Excess"]["Median_Ref1"] - Median_Mix):
                population_median = methods["Excess"]["Median_Ref2"]
            else:  # Ref1 is closets to the mixture
                population_median = methods["Excess"]["Median_Ref1"]
            number_Ref2_low = len(RM[RM <= kwargs['population_median']])
            number_Ref2_high = len(RM[RM > kwargs['population_median']])
            sample_size = len(RM)
    #        proportion_Ref1 = (number_high - number_low)/sample_size
#            print("Passed median:", kwargs['population_median'])
#            print("Ref1 median:", np.median(Ref1))
#            print("Ref2 median:", np.median(Ref2))
#            print("Mixture size:", sample_size)
#            if kwargs['population_median'] < np.median(Ref1):
            results['Excess'] = abs(number_Ref2_high - number_Ref2_low)/sample_size #kwargs['sample_size']
#                print("M_Ref2 < M_Ref1")
#                print("High", number_Ref2_high)
#                print("Low", number_Ref2_low)
#            else:
                # NOTE: This is an extension of the original method (above)
#                results['Excess'] = (number_Ref2_low - number_Ref2_high)/sample_size
#                print("M_Ref1 < M_Ref2")
#                print("High", number_Ref2_high)
#                print("Low", number_Ref2_low)

#            results['Excess'] /= 0.92  # adjusted for fact it underestimates by 8%
            results['Excess'] *= methods["Excess"]["adjustment_factor"]

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

#        run_method = defaultdict(bool)
#        run_method[method] = True

        results = estimate_Ref1(Mix, Ref1, Ref2, method, **extra_args)

        return results#[method]


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
        # with Parallel(n_jobs=nprocs) as parallel:
            # Parallelise over bootstraps

        for method in methods:
            # Fix estimated proportions for each method
            if true_prop_Ref1:
                prop_Ref1 = true_prop_Ref1
            else:
                prop_Ref1 = initial_results[method]
            individual_method = {}
            individual_method[method] = methods[method]
            results[method] = [bootstrap_mixture(sample_size, prop_Ref1, Ref1, Ref2, individual_method, **extra_args)[method]
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
