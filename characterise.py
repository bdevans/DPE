#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

Parallelised script for systematically characterising the parameter space for
proportion estimation methods by generating artificial mixtures across a grid
of proprtions and sample sizes.

@author: ben

Original Diabetes data was processed with commit 01a9705b
https://git.exeter.ac.uk/bdevans/DPE/commit/01a9705b6fa1bf0d1df4fd3a4beaa1a413f64cb5
"""

import os
import time
# import math
# import sys
# import multiprocessing
from collections import OrderedDict

# import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import lmfit
# import dill
from joblib import Parallel, delayed, cpu_count
# from joblib import Memory
# mem = Memory(cachedir='/tmp')
import tqdm

import proportion_estimation as pe
from datasets import (load_diabetes_data, load_renal_data)

# ---------------------------- Define constants ------------------------------

verbose = False
out_dir = "results"
run_means = True
run_excess = True
adjust_excess = False
run_KDE = True
run_EMD = True
check_EMD = False

seed = 42

mixtures = 10  # 1000
bootstraps = 10
sample_sizes = np.arange(100, 2501, 100)  # 3100
# sample_sizes = np.linspace(100, 2500, 25, endpoint=True, dtype=int)
# proportions = np.arange(0.0, 1.01, 0.01)  # Ref1 propoertions
proportions = np.linspace(0.0, 1.0, 101, endpoint=True)

KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# ----------------------------------------------------------------------------


def SecToStr(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u'%d:%02d:%02d' % (h, m, s)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if run_KDE:
    # Define the KDE models
    # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
    def kde_Ref1(x, amp_Ref1=1):
        return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

    def kde_Ref2(x, amp_Ref2=1):
        return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))

    model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

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


def assess_performance(sample_size, prop_Ref1, Ref1, Ref2, methods, bootstraps, **extra_args):
    '''New method using analyse_mixture'''

    assert(0.0 <= prop_Ref1 <= 1.0)
    n_Ref1 = int(round(sample_size * prop_Ref1))
    n_Ref2 = sample_size - n_Ref1

    # Construct mixture
    mixture = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                              np.random.choice(Ref2, n_Ref2, replace=True)))

    scores = {'Ref1': Ref1, 'Ref2': Ref2}
    scores['Mix'] = mixture
    logfile = 'pe_s{}_p{}.log'.format(sample_size, prop_Ref1)
    results = pe.analyse_mixture(scores, bins, methods, bootstraps=bootstraps,
                                 sample_size=-1, alpha=0.05,
                                 true_prop_Ref1=prop_Ref1, verbose=0,
                                 logfile=logfile)

    return results


def construct_mixture(sample_size, prop_Ref1, Ref1, Ref2, methods, **extra_args):

    assert(0.0 <= prop_Ref1 <= 1.0)
    n_Ref1 = int(round(sample_size * prop_Ref1))
    n_Ref2 = sample_size - n_Ref1

    # Construct mixture
    mixture = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                              np.random.choice(Ref2, n_Ref2, replace=True)))

    results = estimate_Ref1(mixture, Ref1, Ref2, methods, **extra_args)

    return results


def estimate_Ref1(RM, Ref1, Ref2, methods, **kwargs):
    '''Estimate the proportion of two reference populations in an unknown mixture.
    The returned proportions are with respect to Ref 1. The proportion of Ref 2 is assumed to be 1 pr(Ref1). '''

    # TODO: Import this from proportion_estimation

    bins = kwargs['bins']
    results = {}

    # ----------------------------- Excess method ----------------------------
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
        i_CDF_Mix = pe.interpolate_CDF(RM, bins['centers'], bins['min'], bins['max'])
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
        results['KDE'] = fit_KDE_model(RM, bins, model, kwargs['initial_params'], kwargs['KDE_kernel'])

    return results


if __name__ == '__main__':

    #nprocs = multiprocessing.cpu_count()
    nprocs = cpu_count()
    #print('Running with {}:{} processors...'.format(nprocs, cpu_count()))
    print('Running with {} processors...'.format(nprocs))

    # Set random seed
    np.random.seed(seed)
    # rng = np.random.RandomState(42)

    np.seterr(divide='ignore', invalid='ignore')

    datasets = {}
    datasets["Renal"] = load_renal_data()
    metric = "T1GRS"
    datasets["Diabetes"] = load_diabetes_data(metric)

    for tag, data in datasets.items():

        print("Running parameter sweep with {} scores...".format(tag))

        t = time.time()  # Start timer

        # Unpack data
        scores, bins, means, medians, prop_Ref1 = data

        Ref1 = scores['Ref1']
        Ref2 = scores['Ref2']
        Mix = scores['Mix']
        bin_width = bins['width']
        bin_edges = bins['edges']

        if adjust_excess:
            adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
        else:
            adjustment_factor = 1.0

        methods = OrderedDict([("Excess", {"Median_Ref1": medians["Ref1"],
                                           "Median_Ref2": medians["Ref2"],
                                           "adjustment_factor": adjustment_factor}),
                               ("Means", {'Ref1': means['Ref1'],
                                          'Ref2': means['Ref2']}),
                               ("EMD", True),
                               ("KDE", {'kernel': KDE_kernel,
                                        'bandwidth': bins['width']})])

        extra_args = {}
#        if sample_size == -1:
#            sample_size = len(Mix)
        extra_args['bins'] = bins

        if "Excess" in methods:
            # --------------------------- Excess method --------------------------
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

            extra_args['population_median'] = median

            if verbose:
                print("Ref1 median:", np.median(Ref1))
                print("Ref2 median:", np.median(Ref2))
                print("Population median: {}".format(median))
#                print("Mixture size:", sample_size)


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


        if "EMD" in methods:
            # ---------------------------- EMD method ----------------------------

            max_EMD = bin_edges[-1] - bin_edges[0]

            # Interpolate the cdfs at the same points for comparison
            i_CDF_Ref1 = pe.interpolate_CDF(Ref1, bins['centers'], bins['min'], bins['max'])
            i_CDF_Ref2 = pe.interpolate_CDF(Ref2, bins['centers'], bins['min'], bins['max'])

            # EMDs computed with interpolated CDFs
            i_EMD_1_2 = sum(abs(i_CDF_Ref1-i_CDF_Ref2))
    #        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD

            extra_args['max_EMD'] = max_EMD
            extra_args['i_CDF_Ref1'] = i_CDF_Ref1
            extra_args['i_CDF_Ref2'] = i_CDF_Ref2
            extra_args['i_EMD_1_2'] = i_EMD_1_2


        if "KDE" in methods:
            # -------------------------- KDE method --------------------------

            labels = ['Ref1', 'Ref2']  # Reference populations

            bw = bin_width  # Bandwidth
            # Fit reference populations
            kdes = pe.fit_kernels(scores, bw)

            try:
                KDE_kernel = methods["KDE"]["kernel"]
            except (KeyError, TypeError):
                if verbose:
                    print("No kernel specified!")
                KDE_kernel = "gaussian"  # Default kernel
            else:
                try:
                    bw = methods["KDE"]["bandwidth"]
                except (KeyError, TypeError):
                    bw = bins["width"]
            finally:
                if verbose:
                    print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))

            # Define the KDE models
            # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
#            def kde_Ref1(x, amp_Ref1):
#                return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))
#
#            def kde_Ref2(x, amp_Ref2):
#                return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))
#
#            model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

            params_mix = model.make_params()
            params_mix['amp_Ref1'].value = 1
            params_mix['amp_Ref1'].min = 0
            params_mix['amp_Ref2'].value = 1
            params_mix['amp_Ref2'].min = 0

#            extra_args['model'] = model  # This breaks joblib
            extra_args['initial_params'] = params_mix
            extra_args['KDE_kernel'] = KDE_kernel
            extra_args['bin_width'] = bin_width
            extra_args['kdes'] = kdes


# AttributeError: Can't pickle local object 'prepare_methods.<locals>.kde_Ref1'
#        extra_args = pe.prepare_methods(methods, scores, bins, verbose=0)

#        if sample_size == -1:
#            sample_size = len(Mix)

        if "Excess" in methods:
            # --------------------------- Excess method --------------------------
            # TODO: Check and rename to Ref1_median?
            results_Excess = np.zeros((len(sample_sizes), len(proportions), mixtures, bootstraps))

        if "Means" in methods:
            # -------------------------- Means method -------------------------
            results_Means = np.zeros((len(sample_sizes), len(proportions), mixtures, bootstraps))

        if "EMD" in methods:
            # ---------------------------- EMD method ----------------------------
            results_EMD = np.zeros((len(sample_sizes), len(proportions), mixtures, bootstraps))

        if "KDE" in methods:
            # -------------------------- KDE method --------------------------
            results_KDE = np.zeros((len(sample_sizes), len(proportions), mixtures, bootstraps))


        # Spawn threads
#        with Parallel(n_jobs=nprocs) as parallel:
        for s in tqdm.trange(len(sample_sizes), desc='Size'):
            sample_size = sample_sizes[s]
            for p in tqdm.trange(len(proportions), desc='p1*', leave=False):
                prop_Ref1 = proportions[p]
                with Parallel(n_jobs=nprocs) as parallel:
                    # Parallelise over mixtures
                    results = parallel(delayed(assess_performance)(sample_size, prop_Ref1, Ref1, Ref2, methods, bootstraps, **extra_args)
                                       for m in range(mixtures))

                    for m in range(mixtures):
                        if run_excess:
                            results_Excess[s, p, m, :] = results[m]['Excess']
                        if run_means:
                            results_Means[s, p, m, :] = results[m]['Means']
                        if run_EMD:
                            results_EMD[s, p, m, :] = results[m]['EMD']
#                            mat_EMD_31[s, p, b] = results[b]['EMD_31']
#                            mat_EMD_32[s, p, b] = results[b]['EMD_32']
                        if run_KDE:
                            results_KDE[s, p, m, :] = results[m]['KDE']


        elapsed = time.time() - t
        print('Elapsed time = {}\n'.format(SecToStr(elapsed)))

        # Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
#        if run_EMD:
#            norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
#            norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21
#            if check_EMD:
#                norm_EMD_dev = emd_dev_from_fit * bin_width / max_emd / i_EMD_21
#                median_error = 100 * np.median(norm_EMD_dev, axis=2)  # Percentage
        if run_excess:
            np.save('{}/excess_{}'.format(out_dir, tag), results_Excess)
        if run_means:
            np.save('{}/means_{}'.format(out_dir, tag), results_Means)
        if run_EMD:
            np.save('{}/emd_{}'.format(out_dir, tag), results_EMD)
#            np.save('{}/emd_31_{}'.format(out_dir, tag), norm_mat_EMD_31)
#            np.save('{}/emd_32_{}'.format(out_dir, tag), norm_mat_EMD_32)
        if run_KDE:
            np.save('{}/kde_{}'.format(out_dir, tag), results_KDE)
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    print("Analysis of methods on datasets: {} complete!".format(list(datasets)))



if False:  # __name__ == '__main__':

    #nprocs = multiprocessing.cpu_count()
    nprocs = cpu_count()
    #print('Running with {}:{} processors...'.format(nprocs, cpu_count()))
    print('Running with {} processors...'.format(nprocs))

    # Set random seed
    np.random.seed(seed)
    # rng = np.random.RandomState(42)

    np.seterr(divide='ignore', invalid='ignore')

    datasets = {}
    datasets["Renal"] = load_renal_data()
    metric = "T1GRS"
    datasets["Diabetes"] = load_diabetes_data(metric)

    for tag, data in datasets.items():

        print("Running parameter sweep with {} scores...".format(tag))

        t = time.time()  # Start timer

        # Unpack data
        scores, bins, means, medians, prop_Ref1 = data

        Ref1 = scores['Ref1']
        Ref2 = scores['Ref2']
        Mix = scores['Mix']
        bin_width = bins['width']
        bin_edges = bins['edges']

        if adjust_excess:
            adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
        else:
            adjustment_factor = 1.0

        methods = OrderedDict([("Excess", {"Median_Ref1": medians["Ref1"],
                                           "Median_Ref2": medians["Ref2"],
                                           "adjustment_factor": adjustment_factor}),
                               ("Means", {'Ref1': means['Ref1'],
                                          'Ref2': means['Ref2']}),
                               ("EMD", True),
                               ("KDE", {'kernel': KDE_kernel,
                                        'bandwidth': bins['width']})])

        extra_args = {}
#        if sample_size == -1:
#            sample_size = len(Mix)
        extra_args['bins'] = bins

        if "Excess" in methods:
            # --------------------------- Excess method --------------------------
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

            extra_args['population_median'] = median

            if verbose:
                print("Ref1 median:", np.median(Ref1))
                print("Ref2 median:", np.median(Ref2))
                print("Population median: {}".format(median))
#                print("Mixture size:", sample_size)

            results_Excess = np.zeros((len(sample_sizes), len(proportions), mixtures))

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

            results_Means = np.zeros((len(sample_sizes), len(proportions), mixtures))

        if "EMD" in methods:
            # ---------------------------- EMD method ----------------------------

            max_EMD = bin_edges[-1] - bin_edges[0]

            # Interpolate the cdfs at the same points for comparison
            i_CDF_Ref1 = pe.interpolate_CDF(Ref1, bins['centers'], bins['min'], bins['max'])
            i_CDF_Ref2 = pe.interpolate_CDF(Ref2, bins['centers'], bins['min'], bins['max'])

            # EMDs computed with interpolated CDFs
            i_EMD_1_2 = sum(abs(i_CDF_Ref1-i_CDF_Ref2))
    #        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD

            extra_args['max_EMD'] = max_EMD
            extra_args['i_CDF_Ref1'] = i_CDF_Ref1
            extra_args['i_CDF_Ref2'] = i_CDF_Ref2
            extra_args['i_EMD_1_2'] = i_EMD_1_2

            results_EMD = np.zeros((len(sample_sizes), len(proportions), mixtures))

        if "KDE" in methods:
            # -------------------------- KDE method --------------------------

            labels = ['Ref1', 'Ref2']  # Reference populations

            bw = bin_width  # Bandwidth
            # Fit reference populations
            kdes = pe.fit_kernels(scores, bw)

            try:
                KDE_kernel = methods["KDE"]["kernel"]
            except (KeyError, TypeError):
                if verbose:
                    print("No kernel specified!")
                KDE_kernel = "gaussian"  # Default kernel
            else:
                try:
                    bw = methods["KDE"]["bandwidth"]
                except (KeyError, TypeError):
                    bw = bins["width"]
            finally:
                if verbose:
                    print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))

            # Define the KDE models
            # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
#            def kde_Ref1(x, amp_Ref1):
#                return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))
#
#            def kde_Ref2(x, amp_Ref2):
#                return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))
#
#            model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

            params_mix = model.make_params()
            params_mix['amp_Ref1'].value = 1
            params_mix['amp_Ref1'].min = 0
            params_mix['amp_Ref2'].value = 1
            params_mix['amp_Ref2'].min = 0

#            extra_args['model'] = model  # This breaks joblib
            extra_args['initial_params'] = params_mix
            extra_args['KDE_kernel'] = KDE_kernel
            extra_args['bin_width'] = bin_width
            extra_args['kdes'] = kdes

            results_KDE = np.zeros((len(sample_sizes), len(proportions), mixtures))

        # Setup progress bar
        iterations = len(sample_sizes) * len(proportions) # * len(metrics) * mixtures  #KDE_fits.size
        max_bars = 78    # number of dots in progress bar
        if iterations < max_bars:
            max_bars = iterations   # if less than 20 points, shorten bar
        print("|" + max_bars*"-" + "|")
        print('|', end='', flush=True)  # print start of progress bar
        it = 0
        bar_element = 0

        # Spawn threads
#        with Parallel(n_jobs=nprocs) as parallel:
        for s, sample_size in enumerate(sample_sizes):
            for p, prop_Ref1 in enumerate(proportions):

#        for s, sample_size in enumerate(tqdm.tqdm(sample_sizes)):
#            for p, prop_Ref1 in enumerate(tqdm.tqdm(proportions)):
                with Parallel(n_jobs=nprocs) as parallel:
                    # Parallelise over mixtures
                    results = parallel(delayed(construct_mixture)(sample_size, prop_Ref1, Ref1, Ref2, methods, **extra_args)
                                       for b in range(mixtures))
#                    results = [construct_mixture(sample_size, prop_Ref1, Ref1, Ref2, methods, **extra_args)
#                                       for b in range(mixtures)]

                    for b in range(mixtures):
                        if run_excess:
                            results_Excess[s, p, b] = results[b]['Excess']
                        if run_means:
                            results_Means[s, p, b] = results[b]['Means']
                        if run_EMD:
                            results_EMD[s, p, b] = results[b]['EMD']
#                            mat_EMD_31[s, p, b] = results[b]['EMD_31']
#                            mat_EMD_32[s, p, b] = results[b]['EMD_32']
                        if run_KDE:
                            results_KDE[s, p, b] = results[b]['KDE']

                    if (it >= bar_element*iterations/max_bars):
                        print('*', end='', flush=True)
                        bar_element += 1
                    if (it >= iterations-1):
                        print('|', flush=True)
                    it += 1

        elapsed = time.time() - t
        print('Elapsed time = {}\n'.format(SecToStr(elapsed)))

        # Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
#        if run_EMD:
#            norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
#            norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21
#            if check_EMD:
#                norm_EMD_dev = emd_dev_from_fit * bin_width / max_emd / i_EMD_21
#                median_error = 100 * np.median(norm_EMD_dev, axis=2)  # Percentage
        if run_excess:
            np.save('{}/excess_{}'.format(out_dir, tag), results_Excess)
        if run_means:
            np.save('{}/means_{}'.format(out_dir, tag), results_Means)
        if run_EMD:
            np.save('{}/emd_{}'.format(out_dir, tag), results_EMD)
#            np.save('{}/emd_31_{}'.format(out_dir, tag), norm_mat_EMD_31)
#            np.save('{}/emd_32_{}'.format(out_dir, tag), norm_mat_EMD_32)
        if run_KDE:
            np.save('{}/kde_{}'.format(out_dir, tag), results_KDE)
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    # run_mixture('T1GRS', scores, medians, bins, sample_sizes, proportions, mixtures)
    # run_mixture('T2GRS', scores, medians, bins, sample_sizes, proportions, mixtures)

    print("Analysis of methods on datasets: {} complete!".format(list(datasets)))
