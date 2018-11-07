#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

Module to analyse an unknown mixture population.

@author: ben
"""

# from collections import OrderedDict  # , defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
# from statsmodels.nonparametric.kde import KDEUnivariate
# TODO: replace with a scipy/numpy function to reduce dependencies
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
import lmfit
# TODO: Try replacing with scipy.optimize.curve_fit to solve joblib problem and reduce dependencies:
# https://lmfit.github.io/lmfit-py/model.html
# from joblib import Parallel, delayed, cpu_count
# import seaborn as sns
# import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
# from tqdm import tqdm
import tqdm
from statsmodels.stats.proportion import proportion_confint

METHODS_ORDER = ["Excess", "Means", "EMD", "KDE"]


def fit_kernels(scores, bw):
    kernels = {}
    for label, data in scores.items():
        kernels[label] = {}
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            # kde = KernelDensity(kernel=kernel, bandwidth=bw, atol=0, rtol=1e-4).fit(X)  #
            # kernels[label][kernel] = kde

            kernels[label][kernel] = KernelDensity(kernel=kernel, bandwidth=bw,
                                                   atol=0, rtol=1e-4).fit(X)
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
    res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])),
                        x=x_KDE, params=params_mix)
    amp_Ref1 = res_mix.params['amp_Ref1'].value
    amp_Ref2 = res_mix.params['amp_Ref2'].value
    return amp_Ref1/(amp_Ref1+amp_Ref2)


def interpolate_CDF(scores, x_i, min_edge, max_edge):
    # TODO: x = [x_i[0], *sorted(scores), x_i[-1]]
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def prepare_methods(methods, scores, bins, verbose=1):

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']
    bin_width = bins['width']
    bin_edges = bins['edges']

    kwargs = {}
    # if sample_size == -1:
    #     sample_size = len(Mix)
    kwargs['bins'] = bins

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
            kwargs['adjustment_factor'] = methods["Excess"]["adjustment_factor"]
    #        if isinstance(methods["Excess"], float):
    #            # Median passed
    #            median = methods["Excess"]
    #            print("Passed median: {}".format(median))
    #        else:
    #            # The Excess method assumes that...
    #            median = np.median(scores["Ref2"])
        kwargs['population_median'] = median
        if verbose > 1:
            print("Ref1 median:", np.median(Ref1))
            print("Ref2 median:", np.median(Ref2))
            print("Population median: {}".format(median))
            print("Mixture size:", len(Mix))  # sample_size)

    if "Means" in methods:
        try:
            Mean_Ref1 = methods["Means"]["Ref1"]
        except (KeyError, TypeError):
            if verbose > 1:
                print("No Mean_Ref1 specified!")
            Mean_Ref1 = Ref1.mean()
        finally:
            kwargs["Mean_Ref1"] = Mean_Ref1
        try:
            Mean_Ref2 = methods["Means"]["Ref2"]
        except (KeyError, TypeError):
            if verbose > 1:
                print("No Mean_Ref2 specified!")
            Mean_Ref2 = Ref2.mean()
        finally:
            kwargs["Mean_Ref2"] = Mean_Ref2

    if "EMD" in methods:
    # -------------------------------- EMD method --------------------------------

        if 'max_EMD' not in kwargs:
            max_EMD = bin_edges[-1] - bin_edges[0]
            kwargs['max_EMD'] = max_EMD

        # Interpolate the cdfs at the same points for comparison
        if 'i_CDF_Ref1' not in kwargs:
            i_CDF_Ref1 = interpolate_CDF(Ref1, bins['centers'], bins['min'], bins['max'])
            kwargs['i_CDF_Ref1'] = i_CDF_Ref1

        if 'i_CDF_Ref2' not in kwargs:
            i_CDF_Ref2 = interpolate_CDF(Ref2, bins['centers'], bins['min'], bins['max'])
            kwargs['i_CDF_Ref2'] = i_CDF_Ref2
    #        i_CDF_Mix = interpolate_CDF(Mix, bin_centers, bins['min'], bins['max'])

        # EMDs computed with interpolated CDFs
        if 'i_EMD_1_2' not in kwargs:
            i_EMD_1_2 = sum(abs(kwargs['i_CDF_Ref1']-kwargs['i_CDF_Ref2']))
            kwargs['i_EMD_1_2'] = i_EMD_1_2
    #        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD

    #        kwargs['i_EMD_21'] = i_EMD_21

    if "KDE" in methods:
        # ------------------------------ KDE method ------------------------------

        bw = bin_width  # Bandwidth

        if 'kdes' not in kwargs:
            kdes = fit_kernels(scores, bw)
            kwargs['kdes'] = kdes
        else:
            kdes = kwargs['kdes']

        if 'KDE_kernel' not in kwargs or 'bin_width' not in kwargs:
            try:
                KDE_kernel = methods["KDE"]["kernel"]
            except (KeyError, TypeError):
                if verbose > 1:
                    print("No kernel specified!")
                KDE_kernel = "gaussian"  # Default kernel
            else:
                try:
                    bw = methods["KDE"]["bandwidth"]
                except (KeyError, TypeError):
                    bw = bins["width"]
            finally:
                if verbose > 1:
                    print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))
            kwargs['KDE_kernel'] = KDE_kernel
            kwargs['bin_width'] = bin_width
        else:
            KDE_kernel = kwargs['KDE_kernel']

        if 'model' not in kwargs:
            # Define the KDE models
            # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
            # Assigning a default value to amp initialises them
            def kde_Ref1(x, amp_Ref1=1):
                return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

            def kde_Ref2(x, amp_Ref2=1):
                return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))

            model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)
            kwargs['model'] = model  # This breaks joblib

        if 'initial_params' not in kwargs:
            params_mix = kwargs['model'].make_params()
            params_mix['amp_Ref1'].value = 1
            params_mix['amp_Ref1'].min = 0
            params_mix['amp_Ref2'].value = 1
            params_mix['amp_Ref2'].min = 0
            kwargs['initial_params'] = params_mix

    return kwargs


def analyse_mixture(scores, bins, methods, bootstraps=1000, sample_size=-1,
                    alpha=0.05, true_prop_Ref1=None, n_jobs=1, seed=None,
                    verbose=1, logfile=''):
                    # , means=None, median=None, KDE_kernel='gaussian'):

    if seed is not None:
        np.random.seed(seed)

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']
    # kwargs["fit_KDE_model"] = fit_KDE_model
    kwargs = prepare_methods(methods, scores, bins, verbose=verbose)
    # print(kwargs)

    def estimate_Ref1(RM, Ref1, Ref2, methods, **kwargs):
        '''Estimate the proportion of two reference populations in an unknown mixture.
        The returned proportions are with respect to Ref 1. The proportion of Ref 2 is assumed to be 1 pr(Ref1). '''

        bins = kwargs['bins']
        results = {}

        # -------------------------- Subtraction method --------------------------
        if "Excess" in methods:
            # Calculate the proportion of another population w.r.t. the excess
            # number of cases from the mixture's assumed majority population.
            # TODO: Flip these around for when using the T2GRS
            # Median_Mix = np.median(Mix)

            # TODO!!!
            # if abs(methods["Excess"]["Median_Ref2"] - Median_Mix) < abs(methods["Excess"]["Median_Ref1"] - Median_Mix):
            #     population_median = methods["Excess"]["Median_Ref2"]
            # else:  # Ref1 is closets to the mixture
            #     population_median = methods["Excess"]["Median_Ref1"]

            # Ref1: disease; Ref2: healthy
#            if medians["Ref1"] > medians["Ref2"]:
#                excess_cases = RM[RM > kwargs['population_median']].count()
#                expected_cases = RM[RM <= kwargs['population_median']].count()
#            else:  # disease has a lower median than the healthy population
#                excess_cases = RM[RM <= kwargs['population_median']].count()
#                expected_cases = RM[RM > kwargs['population_median']].count()
#            sample_size = len(RM)
#            results['Excess'] = abs(excess_cases - expected_cases)/sample_size


            number_low = len(RM[RM <= kwargs['population_median']])
            number_high = len(RM[RM > kwargs['population_median']])
            sample_size = len(RM)
    #        proportion_Ref1 = (number_high - number_low)/sample_size
#            print("Passed median:", kwargs['population_median'])
#            print("Ref1 median:", np.median(Ref1))
#            print("Ref2 median:", np.median(Ref2))
#            print("Mixture size:", sample_size)
#            if kwargs['population_median'] < np.median(Ref1):
            results['Excess'] = abs(number_high - number_low)/sample_size #kwargs['sample_size']
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

            # Clip results
            if results['Excess'] > 1:
                results['Excess'] = 1.0
            if results['Excess'] < 0:
                results['Excess'] = 0.0

        # ---------------------- Difference of Means method ----------------------
        if "Means" in methods:
            # proportion_of_Ref1 = (RM.mean()-kwargs['Mean_Ref2'])/(kwargs['Mean_Ref1']-kwargs['Mean_Ref2'])
            # results['Means'] = abs(proportion_of_Ref1)

            # TODO!!!
            if kwargs['Mean_Ref1'] > kwargs['Mean_Ref2']:
                proportion_of_Ref1 = (RM.mean()-kwargs['Mean_Ref2'])/(kwargs['Mean_Ref1']-kwargs['Mean_Ref2'])
            else:
                proportion_of_Ref1 = (kwargs['Mean_Ref2']-RM.mean())/(kwargs['Mean_Ref2']-kwargs['Mean_Ref1'])
            if proportion_of_Ref1 < 0:
                proportion_of_Ref1 = 0.0
            if proportion_of_Ref1 > 1:
                proportion_of_Ref1 = 1.0
            results['Means'] = proportion_of_Ref1

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

            i_CDF_Ref1 = kwargs["i_CDF_Ref1"]
            i_CDF_Ref2 = kwargs["i_CDF_Ref2"]
            i_EMD_1_2 = kwargs["i_EMD_1_2"]
            i_EMD_M_1 = sum(abs(i_CDF_Mix-i_CDF_Ref1))
            i_EMD_M_2 = sum(abs(i_CDF_Mix-i_CDF_Ref2))
#            i_EMD_1_2 = sum(abs(i_CDF_Ref1-i_CDF_Ref2))
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

        # ------------------------------ KDE method ------------------------------
        if "KDE" in methods:
            # TODO: Print out warnings if goodness of fit is poor?
            results['KDE'] = fit_KDE_model(RM, bins, kwargs['model'],
                                           kwargs['initial_params'],
                                           kwargs['KDE_kernel'])

        return results


#    def construct_bootstraps(sample_size, prop_Ref1, Ref1, Ref2, methods, **kwargs):
#
#        results = {}
#        for method in methods:
#            assert(0.0 <= prop_Ref1[method] <= 1.0)
#            n_Ref1 = int(round(sample_size * prop_Ref1[method]))
#            n_Ref2 = sample_size - n_Ref1
#
#            # Bootstrap mixture
#            Mix = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
#                                  np.random.choice(Ref2, n_Ref2, replace=True)))
#
##            run_method = defaultdict(bool)
##            run_method[method] = True
#            indiv_method = {}
#            indiv_method[method] = methods[method]
#
#            results[method] = estimate_Ref1(Mix, Ref1, Ref2, indiv_method, **kwargs)[method]
#
#        return results#[method]


    def bootstrap_mixture(Mix, Ref1, Ref2, methods, sample_size=-1, seed=None, **kwargs):

#        results = {}
#        for method in methods:
#            # Bootstrap mixture
#            bs = np.random.choice(Mix, sample_size, replace=True)
#
#            indiv_method = {}
#            indiv_method[method] = methods[method]
#
#            results[method] = estimate_Ref1(bs, Ref1, Ref2, indiv_method, **kwargs)[method]

        if sample_size == -1:
            sample_size = len(Mix)

        bs = np.random.RandomState(seed).choice(Mix, sample_size, replace=True)
        # bs = np.random.choice(Mix, sample_size, replace=True)
        results = estimate_Ref1(bs, Ref1, Ref2, methods, **kwargs)

        return results

    columns = [method for method in METHODS_ORDER if method in methods]

    if bootstraps <= 1:
        # Get initial estimate of proportions
        initial_results = estimate_Ref1(Mix, Ref1, Ref2, methods, **kwargs)
        if verbose > 1:
            pprint(initial_results)
        if true_prop_Ref1:
            if verbose > 1:
                print("Ground truth: {:.5f}".format(true_prop_Ref1))

        df_pe = pd.DataFrame(initial_results, index=[0], columns=columns)

    else:  # if bootstraps > 1:
        if verbose > 0:
            if n_jobs == -1:
                nprocs = cpu_count()
            else:
                nprocs = n_jobs
            print('Running {} bootstraps with {} processors...'.format(bootstraps, nprocs), flush=True)

        # results = OrderedDict()
        # Make bootstrapping deterministic with parallelism
        # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
        boot_seeds = np.random.randint(np.iinfo(np.int32).max, size=bootstraps)

        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(bootstrap_mixture)(Mix, Ref1, Ref2, methods, sample_size, seed=seed, **kwargs)
                               for seed in tqdm.tqdm(boot_seeds, desc="Bootstraps", dynamic_ncols=True, disable=disable))  # , leave=False
                               # for b in tqdm.trange(bootstraps, desc="Bootstraps", dynamic_ncols=True, disable=disable))

        # NOTE: These results are identical to when n_jobs=1 however it takes about 25% less time per iteration
        # results = [bootstrap_mixture(Mix, Ref1, Ref2, methods, sample_size, **kwargs)
        #            for b in tqdm.trange(bootstraps, desc="Bootstraps", dynamic_ncols=True, disable=disable)]

#        # results = OrderedDict()
#
##        for method in methods:
#            # Fix estimated proportions for each method
##            if true_prop_Ref1:
##                prop_Ref1 = defaultdict(lambda: true_prop_Ref1)
###                prop_Ref1[method] = true_prop_Ref1
##            else:
##                prop_Ref1 = initial_results
##            individual_method = {}
##            individual_method[method] = methods[method]
##            results[method] = [bootstrap_mixture(sample_size, prop_Ref1, Ref1, Ref2, individual_method, **kwargs)[method]
##                               for b in range(bootstraps)]
#
#            results = [bootstrap_mixture(Mix, Ref1, Ref2, methods, sample_size, **kwargs)
#                       for b in tqdm.trange(bootstraps, ncols=100, desc="Bootstraps")]
#        else:  # Disable progress bar
#            with Parallel(n_jobs=n_jobs) as parallel:
#                results = parallel(delayed(bootstrap_mixture)(Mix, Ref1, Ref2, methods, sample_size, **kwargs)
#                                   for b in range(bootstraps))
##            results = [bootstrap_mixture(Mix, Ref1, Ref2, methods, sample_size, **kwargs)
##                       for b in range(bootstraps)]

        # Put into dataframe
        df_pe = pd.DataFrame.from_records(results, columns=columns)

    if verbose:
        # ------------ Summarise proportions for the whole distribution --------------
        print()
        print("{:20} | {:^17s} | {:^17s} ".format("Proportion Estimates", "Reference 1", "Reference 2"))
        print("="*61)
        for method in methods:
    #        print("{:20} | {:<17.5f} | {:<17.5f} ".format(method, initial_results[method], 1-initial_results[method]))
            print("{:14} (µ±σ) | {:.5f} +/- {:.3f} | {:.5f} +/- {:.3f} ".format(method, np.mean(df_pe[method]), np.std(df_pe[method]), 1-np.mean(df_pe[method]), np.std(1-df_pe[method])))  #  (+/- SD)
            if bootstraps > 1:
                nobs = len(df_pe[method])
                count = int(np.mean(df_pe[method])*nobs)
                ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method='normal')
                ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')
                print("C.I. (level={:3.1%})   | {:.5f},  {:.5f} | {:.5f},  {:.5f} ".format(1-alpha, ci_low1, ci_upp1, ci_low2, ci_upp2))
    #            print("{:20} | {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f} |".format("C.I. (level=95%)", ci_low1, np.mean(df_pe[method]), ci_upp1, ci_low2, 1-np.mean(df_pe[method]), ci_upp2))
            print("-"*61)
        if true_prop_Ref1:
            print("{:20} | {:<17.5f} | {:<17.5f}".format("Ground Truth", true_prop_Ref1, 1-true_prop_Ref1))
            print("="*61)
        print()

    if logfile == '':
        logfile = "proportion_estimate.log"
    with open(logfile, 'w') as lf:
        lf.write("{:20} | {:^17s} | {:^17s} \n".format("Proportion Estimates", "Reference 1", "Reference 2"))
        lf.write("="*61)
        lf.write("\n")
        for method in methods:
    #        print("{:20} | {:<17.5f} | {:<17.5f} ".format(method, initial_results[method], 1-initial_results[method]))
            lf.write("{:14} (µ±σ) | {:.5f} +/- {:.3f} | {:.5f} +/- {:.3f} \n".format(method, np.mean(df_pe[method]), np.std(df_pe[method]), 1-np.mean(df_pe[method]), np.std(1-df_pe[method])))  #  (+/- SD)
            if bootstraps > 1:
                nobs = len(df_pe[method])
                count = int(np.mean(df_pe[method])*nobs)
                ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method='normal')
                ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')
                lf.write("C.I. (level={:3.1%})   | {:.5f},  {:.5f} | {:.5f},  {:.5f} \n".format(1-alpha, ci_low1, ci_upp1, ci_low2, ci_upp2))
    #            print("{:20} | {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f} |".format("C.I. (level=95%)", ci_low1, np.mean(df_pe[method]), ci_upp1, ci_low2, 1-np.mean(df_pe[method]), ci_upp2))
            lf.write("-"*61)
            lf.write("\n")
        if true_prop_Ref1:
            lf.write("{:20} | {:<17.5f} | {:<17.5f}\n".format("Ground Truth", true_prop_Ref1, 1-true_prop_Ref1))
            lf.write("="*61)
            lf.write("\n")
        lf.write("\n")

    return df_pe
