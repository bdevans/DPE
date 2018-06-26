#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

@author: ben
"""

import os
import time
import math
import sys
from collections import defaultdict, OrderedDict

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.neighbors import KernelDensity
# TODO: replace with a scipy/numpy function to reduce dependencies
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
import lmfit
from joblib import Parallel, delayed, cpu_count


def fit_kernels(scores, bw):
    kernels = {}
    for label, data in scores.items():
        kernels[label] = {}
        X = data[:, np.newaxis]
#        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
            kernels[label][kernel] = kde
    return kernels






# @mem.cache
def fit_KDE(Mix, bins, model, params_mix, kernel):
    x_KDE = np.linspace(bins['min'], bins['max'], len(Mix)+2)
    #x_KDE = np.array([0.095, *np.sort(RM), 0.35])
    mix_kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(Mix[:, np.newaxis])
    res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
    amp_Ref1 = res_mix.params['amp_Ref1'].value
    amp_Ref2 = res_mix.params['amp_Ref2'].value
    return amp_Ref1/(amp_Ref1+amp_Ref2)






def interpolate_CDF(scores, x_i, min_edge, max_edge):
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def analyse_mixture(scores, means, median, bins, run_method, bootstrap=1000):

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']
    population_median = median
    bin_width = bins['width']
    bin_centers = bins['centers']
    bin_edges = bins['edges']

#    print('Running {} mixture analysis...'.format(tag))
#    print('--------------------------------------------------------------------------------')

#    if run_method["Excess"]:
#        # Excess method median of GRS from the whole population in Biobank
#        high = len(Mix[Mix > population_median]) - len(Mix[Mix <= population_median])
#        low = 2*len(Mix[Mix <= population_median])

    # -------------------------------- EMD method --------------------------------

    if run_method["EMD"]:



        max_EMD = bin_edges[-1] - bin_edges[0]

        # Interpolate the cdfs at the same points for comparison
        i_CDF_Ref1 = interpolate_CDF(Ref1, bins['centers'], bins['min'], bins['max'])
        i_CDF_Ref2 = interpolate_CDF(Ref2, bins['centers'], bins['min'], bins['max'])
#        i_CDF_Mix = interpolate_CDF(Mix, bin_centers, bins['min'], bins['max'])

        # EMDs computed with interpolated CDFs
        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
#        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
#        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD


    if run_method["KDE"]:
        # ------------------------------ KDE method ------------------------------

        bw = bin_width  # Bandwidth

        kdes = fit_kernels(scores, bw)
#        kdes = {}
#        labels = ['Reference 1', 'Reference 2', 'Mixture']

        # sp.interpolate.interp1d(X, y, X_new, y_new)

#        for label, data in scores.items():
#            kdes[label] = {}
#            X = data[:, np.newaxis]
#            for kernel in ['gaussian', 'tophat', 'epanechnikov']:
#                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
#                kdes[label][kernel] = kde

        kernel = 'gaussian'  #'epanechnikov'

        # Define the KDE models
        # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
        def kde_Ref1(x, amp_Ref1):
            return amp_Ref1 * np.exp(kdes['Ref1'][kernel].score_samples(x[:, np.newaxis]))


        def kde_Ref2(x, amp_Ref2):
            return amp_Ref2 * np.exp(kdes['Ref2'][kernel].score_samples(x[:, np.newaxis]))


        model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)


        params_mix = model.make_params()
        params_mix['amp_Ref1'].value = 1
        params_mix['amp_Ref1'].min = 0
        params_mix['amp_Ref2'].value = 1
        params_mix['amp_Ref2'].min = 0

#        if plot_results and False:
#            plot_kernels()
#
#        (freqs_Mix, _) = np.histogram(Mix, bins=bins['edges'])
#        x = bins['centers']

    #    res_mix = model.fit(freqs_Mix, x=x, params=params_mix)
    #
    #    dely = res_mix.eval_uncertainty(sigma=3)
    #
    #    amp_T1 = res_mix.params['amp_Ref1'].value
    #    amp_T2 = res_mix.params['amp_Ref2'].value
    #
    #    kde1 = kde_T1(x, amp_T1)
    #    kde2 = kde_T2(x, amp_T2)

#        res_mix = model.fit(freqs_Mix, x=x, params=params_mix)
#
#        dely = res_mix.eval_uncertainty(sigma=3)
#
#        amp_Ref1 = res_mix.params['amp_Ref1'].value
#        amp_Ref2 = res_mix.params['amp_Ref2'].value
#
#        kde1 = kde_Ref1(x, amp_Ref1)
#        kde2 = kde_Ref2(x, amp_Ref2)
#
#        if plot_results:
#            plt.figure()
#            fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False)
#
#            axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=labels[:-1])
#            legend = axP.legend(facecolor='grey')
#            #legend.get_frame().set_facecolor('grey')
#            axP.set_title('Proportions of Type 1 and Type 2 vs {}'.format(tag))
#
#            plt.sca(axM)
#            res_mix.plot_fit()
#
#            axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")
#
#            plt.sca(axR)
#            res_mix.plot_residuals()
#
#            #plt.sca(axI)
#            axI.plot(x, kde1, label='Reference 1')
#            axI.plot(x, kde2, label='Reference 2')
#
#        if verbose:
#            print(res_mix.fit_report())
#            print('Ref2/Ref1 =', amp_Ref2/amp_Ref1)
#            print('')
#            print('\nParameter confidence intervals:')
#            print(res_mix.ci_report())  # --> res_mix.ci_out # See also res_mix.conf_interval()
#



    extra_args = {}

    if run_method["KDE"]:
        extra_args['model'] = model  # This breaks joblib
        extra_args['params_mix'] = params_mix
        extra_args['kernel'] = kernel
        extra_args['bin_width'] = bin_width
        extra_args['kdes'] = kdes
#        extra_args["fit_KDE"] = fit_KDE

    if run_method["EMD"]:
        extra_args['max_EMD'] = max_EMD
        extra_args['i_CDF_Ref1'] = i_CDF_Ref1
        extra_args['i_CDF_Ref2'] = i_CDF_Ref2
        extra_args['i_EMD_21'] = i_EMD_21

    if run_method["Means"]:
        extra_args['Ref1_mean'] = Ref1.mean()
        extra_args['Ref2_mean'] = Ref2.mean()

    if run_method["Excess"]:
        extra_args['population_median'] = median #population_median

    sample_size = len(Mix)
#    extra_args['sample_size'] = sample_size
    extra_args['bins'] = bins
#    extra_args['bin_centers'] = bins['centers']

#    print(extra_args)



    def estimate_Ref1(RM, Ref1, Ref2, run_method, **kwargs):
        '''Estimate the proportion of two reference populations in an unknown mixture.
        The returned proportions are with respect to Ref 1. The proportion of Ref 2 is assumed to be 1 pr(Ref1). '''


        bins = kwargs['bins']
    #    kdes = kwargs['kdes']
        results = {}

        # ---------------------- Difference of Means method ----------------------
        if run_method["Means"]:
            proportion_of_Ref1 = (RM.mean()-kwargs['Ref2_mean'])/(kwargs['Ref1_mean']-kwargs['Ref2_mean'])
            results['Means'] = abs(proportion_of_Ref1)

        # -------------------------- Subtraction method --------------------------
        if run_method["Excess"]:
            number_low = len(RM[RM <= kwargs['population_median']])
            number_high = len(RM[RM > kwargs['population_median']])
            sample_size = len(RM)
    #        proportion_Ref1 = (number_high - number_low)/sample_size
            results['Excess'] = (number_high - number_low)/sample_size #kwargs['sample_size']

        # ------------------------------ KDE method ------------------------------
        if run_method["KDE"]:
            results['KDE'] = fit_KDE(RM, bins, model, kwargs['params_mix'], kwargs['kernel'])

        # ------------------------------ EMD method ------------------------------
        if run_method["EMD"]:
            # Interpolated cdf (to compute EMD)
            i_CDF_Mix = interpolate_CDF(RM, bins['centers'], bins['min'], bins['max'])
    #            x = [bins['min'], *np.sort(RM), bins['max']]
    #            y = np.linspace(0, 1, num=len(x), endpoint=True)
    #            (iv, ii) = np.unique(x, return_index=True)
    #            si_CDF_Mix = np.interp(bin_centers, iv, y[ii])

            # Compute EMDs
            i_EMD_M_1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD #kwargs['max_EMD']
            i_EMD_M_2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD #kwargs['max_EMD']

#            print('M_1', i_EMD_M_1)
#            print('M_2', i_EMD_M_2)
#            print('i_CDF_Mix', i_CDF_Mix)
#            print('max_EMD', max_EMD)
            # TODO: Consider averaging the two measures
            results["EMD"] = 1 - (i_EMD_M_1 / (i_EMD_M_1 + i_EMD_M_2))

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
    initial_results = estimate_Ref1(Mix, Ref1, Ref2, run_method, **extra_args)



    if bootstrap:

        results = OrderedDict()

        methods = [method for method in ["Means", "Excess", "EMD", "KDE"] if run_method[method]]

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
            prop_Ref1 = initial_results[method]
            results[method] = [bootstrap_mixture(sample_size, prop_Ref1, Ref1, Ref2, method, **extra_args)
                                                 for b in range(bootstrap)]
        # Put into dataframe
        df_bs = pd.DataFrame(results)
        #bootstraps[]


    # ------------ Summarise proportions for the whole distribution --------------
    for method in run_method:
        print('Proportions based on {}'.format(method))
        print('Reference 1: {:.5}'.format(initial_results[method]))
        if bootstrap:
            print('Reference 1 mean: {:.5} +/- {:.5}'.format(np.mean(df_bs[method]), np.std(df_bs[method])))
        print('Reference 2: {:.5}'.format(1-initial_results[method]))
        if bootstrap:
            print('Reference 2 mean: {:.5} +/- {:.5}'.format(1-np.mean(df_bs[method]), np.std(1-df_bs[method])))



#    if run_method["Means"]:
#        print('Proportions based on means')
#        print('% of Type 1:', (Mix.mean()-means[tag]['T2'])/(means[tag]['T1']-means[tag]['T2']))
#        print('% of Type 2:', (1-(Mix.mean()-means[tag]['T2'])/(means[tag]['T1']-means[tag]['T2'])))
#
#    if run_method["Excess"]:
#        print('Proportions based on excess')
#        # TODO: Flip these around for when using the T2GRS
#        print('% of Type 1:', (high/(low+high)))
#        print('% of Type 2:', (1-(high/(low+high))))
#
#    print('Proportions based on counts')
#    print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
#    print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))
#
#    if run_method["EMD"]:
#        print("Proportions based on Earth Mover's Distance (histogram values):")
#        print('% of Type 1:', 1-EMD_31/EMD_21)
#        print('% of Type 2:', 1-EMD_32/EMD_21)
#
#        print("Proportions based on Earth Mover's Distance (interpolated values):")
#        print('% of Type 1:', 1-i_EMD_31/i_EMD_21)
#        print('% of Type 2:', 1-i_EMD_32/i_EMD_21)
#
#    if run_method["KDE"]:
#        print('Proportions based on KDEs')
#        print('% of Type 1:', amp_T1/(amp_T1+amp_T2))
#        print('% of Type 2:', amp_T2/(amp_T1+amp_T2))

    print('--------------------------------------------------------------------------------\n\n')

    if bootstrap:
        return (initial_results, df_bs)
    else:
        return (initial_results, None)




if __name__ == '__main__':

#    # Define the KDE models
#    # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
#    def kde_Ref1(x, amp_Ref1):
#        return amp_Ref1 * np.exp(kdes['Ref 1'][kernel].score_samples(x[:, np.newaxis]))
#
#
#    def kde_Ref2(x, amp_Ref2):
#        return amp_Ref2 * np.exp(kdes['Ref 2'][kernel].score_samples(x[:, np.newaxis]))


#    # @mem.cache
#    def fit_KDE(Mix, bins, model, params_mix, kernel):
#        x_KDE = np.linspace(bins['min'], bins['max'], len(Mix)+2)
#        #x_KDE = np.array([0.095, *np.sort(RM), 0.35])
#        mix_kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(Mix[:, np.newaxis])
#        res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
#        amp_Ref1 = res_mix.params['amp_Ref1'].value
#        amp_Ref2 = res_mix.params['amp_Ref2'].value
#        return amp_Ref1/(amp_Ref1+amp_Ref2)
#

    def plot_kernels(scores, bw):
        kernels = ['gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine']
        fig, axes = plt.subplots(len(kernels), 1, sharex=True) #, squeeze=False)
        X_plot = np.linspace(0.1, 0.35, 1000)[:, np.newaxis]

#        for data, label, ax in zip([Ref1, Ref2, Mix], labels, axes):
        for data, label, ax in zip(scores.items(), axes):

            kdes[label] = {}
            X = data[:, np.newaxis]

            for kernel in kernels:
                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
                log_dens = kde.score_samples(X_plot)
                ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                        label="kernel = '{0}'; bandwidth = {1}".format(kernel, bw))
                kdes[label][kernel] = kde  # np.exp(log_dens)

            # ax.text(6, 0.38, "N={0} points".format(N))
            ax.legend(loc='upper left')
            ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
            ax.set_ylabel(label)

#    model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

# ---------------------------- Define constants ------------------------------

    verbose = False
    plot_results = False
    out_dir = "results_test"

    # TODO: Reimplement this
    adjust_excess = True

    seed = 42
    bootstraps = 1000

    dataset = 'data/biobank_mix_WTCC_ref.csv'
    metrics = ['T1GRS', 'T2GRS']
    headers = {'diabetes_type': 'type', 't1GRS': 'T1GRS', 't2GRS': 'T2GRS'}

    T1GRS_bin_width = 0.005
    T1GRS_bin_min = 0.095
    T1GRS_bin_max = 0.350

    T2GRS_bin_width = 0.1
    T2GRS_bin_min = 4.7
    T2GRS_bin_max = 8.9

    medians = {'T1GRS': 0.23137931525707245, 'T2GRS': 6.78826}

    kernel = 'gaussian'
    # kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# ----------------------------------------------------------------------------

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

#    nprocs = multiprocessing.cpu_count()
    nprocs = cpu_count()
    print('Running with {} processors...'.format(nprocs))

    # Set random seed
    np.random.seed(seed)
    # rng = np.random.RandomState(42)

    np.seterr(divide='ignore', invalid='ignore')

    data = pd.read_csv(dataset)
    data.rename(columns=headers, inplace=True)
    data.describe()

    scores = {}
    means = {}

    # Arrays of T1GRS scores for each group
    scores['T1GRS'] = {'Ref1': data.loc[data['type'] == 1, 'T1GRS'].values,
                       'Ref2': data.loc[data['type'] == 2, 'T1GRS'].values,
                       'Mix': data.loc[data['type'] == 3, 'T1GRS'].values}

    means['T1GRS'] = {'Ref1': scores['T1GRS']['Ref1'].mean(),
                      'Ref2': scores['T1GRS']['Ref2'].mean()}

    # Arrays of T2GRS scores for each group
    scores['T2GRS'] = {'Ref1': data.loc[data['type'] == 1, 'T2GRS'].values,
                       'Ref2': data.loc[data['type'] == 2, 'T2GRS'].values,
                       'Mix': data.loc[data['type'] == 3, 'T2GRS'].values}

    means['T2GRS'] = {'Ref1': scores['T2GRS']['Ref1'].mean(),
                      'Ref2': scores['T2GRS']['Ref2'].mean()}

    # ----------------------------- Bin the data ---------------------------------
    N = data.count()[0]
    bins = {}

    T1GRS_bin_edges = np.arange(T1GRS_bin_min, T1GRS_bin_max+T1GRS_bin_width, T1GRS_bin_width)
    T1GRS_bin_centers = (T1GRS_bin_edges[:-1] + T1GRS_bin_edges[1:]) / 2

    bins['T1GRS'] = {'width': T1GRS_bin_width,
                     'min': T1GRS_bin_min,
                     'max': T1GRS_bin_max,
                     'edges': T1GRS_bin_edges,
                     'centers': T1GRS_bin_centers}
    del T1GRS_bin_width, T1GRS_bin_min, T1GRS_bin_max, T1GRS_bin_edges, T1GRS_bin_centers

    # bin_centers = np.arange(0.095+bin_width/2, 0.35+bin_width/2, bin_width)
    T2GRS_bin_edges = np.arange(T2GRS_bin_min, T2GRS_bin_max+T2GRS_bin_width, T2GRS_bin_width)
    T2GRS_bin_centers = (T2GRS_bin_edges[:-1] + T2GRS_bin_edges[1:]) / 2

    bins['T2GRS'] = {'width': T2GRS_bin_width,
                     'min': T2GRS_bin_min,
                     'max': T2GRS_bin_max,
                     'edges': T2GRS_bin_edges,
                     'centers': T2GRS_bin_centers}
    del T2GRS_bin_width, T2GRS_bin_min, T2GRS_bin_max, T2GRS_bin_edges, T2GRS_bin_centers


    run_method = {"Means": True,
                  "Excess": True,
                  "EMD": True,
                  "KDE": True}

    print("Running mixture analysis with T1GRS scores...")
    t = time.time()  # Start timer

    (res, df_bs) = analyse_mixture(scores['T1GRS'], means['T1GRS'], medians['T1GRS'], bins['T1GRS'], run_method, bootstrap=8)
    #    analyse_mixture('T2GRS', scores, means, medians, bins)

    elapsed = time.time() - t
    print('Elapsed time = {:.3f} seconds\n'.format(elapsed))
