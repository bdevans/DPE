#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

@author: ben

Original Diabetes data was processed with commit 01a9705b
https://git.exeter.ac.uk/bdevans/DPE/commit/01a9705b6fa1bf0d1df4fd3a4beaa1a413f64cb5
"""

import os
import time
import math
import sys
#import multiprocessing
from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import lmfit
#import dill
from joblib import Parallel, delayed, cpu_count
# from joblib import Memory
# mem = Memory(cachedir='/tmp')

import analyse_mixture as pe
from datasets import *

# ---------------------------- Define constants ------------------------------

verbose = False
out_dir = "results"
run_means = True
run_excess = True
adjust_excess = True
run_KDE = True
run_EMD = True
check_EMD = False

seed = 42

bootstraps = 1000
sample_sizes = np.arange(100, 1501, 100)  # 3100
proportions = np.arange(0.0, 1.01, 0.01)  # Ref1 propoertions

kernel = 'gaussian'
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
    def kde_Ref1(x, amp_Ref1):
        return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

    def kde_Ref2(x, amp_Ref2):
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


def estimate_Ref1(sample_size, prop_Ref1, b, **kwargs):

    nRef1 = int(round(sample_size * prop_Ref1))
    nRef2 = sample_size - nRef1

    # Random sample from Ref1
    R1 = np.random.choice(Ref1, nRef1, replace=True)

    # Random sample from Ref2
    R2 = np.random.choice(Ref2, nRef2, replace=True)

    # Bootstrap mixture
    RM = np.concatenate((R1, R2))
    # xRM = np.linspace(0, 1, num=len(RM), endpoint=True)

    # assert sample_size == len(RM)

    # x = np.array([0.095, *np.sort(RM), 0.35])
    results = {}

    # ---------------------- Difference of Means method ----------------------
    if run_means:
        proportion_of_Ref1 = (RM.mean()-Ref2_mean)/(Ref1_mean-Ref2_mean)
        results['means'] = abs(proportion_of_Ref1)

    # -------------------------- Subtraction method --------------------------
    if run_excess:
        number_low = len(RM[RM <= population_median])
        number_high = len(RM[RM > population_median])
        proportion_Ref1 = (number_high - number_low)/sample_size
        results['excess'] = proportion_Ref1

    # ------------------------------ KDE method ------------------------------
    if run_KDE:
        # results['KDE'] = fit_KDE(RM, model, params_mix, kernel, bins)
        results['KDE'] = pe.fit_KDE_model(RM, bins, model, params_mix, kernel)

    # ------------------------------ EMD method ------------------------------
    if run_EMD:
        # Interpolated cdf (to compute EMD)
        x = [bins[tag]['min'], *np.sort(RM), bins[tag]['max']]
        y = np.linspace(0, 1, num=len(x), endpoint=True)
        (iv, ii) = np.unique(x, return_index=True)
        si_CDF_3 = np.interp(bin_centers, iv, y[ii])

        # Compute EMDs
        i_EMD_31 = sum(abs(si_CDF_3-i_CDF_1)) * bin_width / max_emd
        i_EMD_32 = sum(abs(si_CDF_3-i_CDF_2)) * bin_width / max_emd

        if check_EMD:
            # These were computed to check that the EMD computed proportions fit the mixture's CDF
            EMD_diff = si_CDF_3 - ((1-i_EMD_31/i_EMD_21)*i_CDF_1 + (1-i_EMD_32/i_EMD_21)*i_CDF_2)
            emd_dev_from_fit[s, p, b] = sum(EMD_diff)  # deviations from fit measured with emd
            rms_dev_from_fit[s, p, b] = math.sqrt(sum(EMD_diff**2)) / len(si_CDF_3)  # deviations from fit measured with rms

        results['EMD_31'] = i_EMD_31
        results['EMD_32'] = i_EMD_32

    return results


if __name__ == '__main__':

    nprocs = multiprocessing.cpu_count()
    print('Running with {}:{} processors...'.format(nprocs, cpu_count()))

    # Set random seed
    np.random.seed(seed)
    # rng = np.random.RandomState(42)

    np.seterr(divide='ignore', invalid='ignore')

    datasets = {}
    metric = "T1GRS"
    datasets["Diabetes"] = load_diabetes_data(metric)
    datasets["Renal"] = load_renal_data()

    for tag, data in datasets.items():

        print("Running bootstrap with {} scores...".format(tag))

        scores, bins, means, medians, prop_Ref1 = data

        # Setup progress bar
        iterations = len(sample_sizes) * len(proportions) # * len(metrics) * bootstraps  #KDE_fits.size
        max_bars = 78    # number of dots in progress bar
        if iterations < max_bars:
            max_bars = iterations   # if less than 20 points, shorten bar
        print("|" + max_bars*"-" + "|")
        sys.stdout.write('|')
        sys.stdout.flush()  # print start of progress bar

        t = time.time()  # Start timer
        it = 0
        bar_element = 0

        Ref1 = scores[tag]['Ref1']
        Ref2 = scores[tag]['Ref2']

        kwargs = {}
        kwargs[tag] = tag
        kwargs['Ref1'] = Ref1
        kwargs['Ref2'] = Ref2

        bin_width = bins[tag]['width']
        bin_centers = bins[tag]['centers']
        bin_edges = bins[tag]['edges']

        if run_KDE:
            bw = bin_width  # Bandwidth
            kdes = {}
            labels = ['Ref1', 'Ref2']  # Reference populations

            # Fit reference populations
            for data, label in zip([Ref1, Ref2], labels):
                kdes[label] = {}
                X = data[:, np.newaxis]
                # for kernel in ['gaussian', 'tophat', 'epanechnikov']:
                kde = KernelDensity(kernel=kernel, bandwidth=bin_width).fit(X)
                # kdes[label][kernel] = kde
                kdes[label] = kde

            # kernel = 'gaussian'  #' epanechnikov'

            params_mix = model.make_params()
            params_mix['amp_Ref1'].value = 1
            params_mix['amp_Ref2'].value = 1

        #    if run_KDE:
            KDE_fits = np.zeros((len(sample_sizes), len(proportions), bootstraps))

            # kwargs['model'] = model  # This breaks joblib
            kwargs['params_mix'] = params_mix
            kwargs['kernel'] = kernel
            kwargs['bin_width'] = bin_width

        if run_EMD:

            max_emd = bin_edges[-1] - bin_edges[0]

            mat_EMD_31 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            mat_EMD_32 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            if check_EMD:
                emd_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))
                rms_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))

            # Interpolate the cdfs at the same points for comparison
            x_Ref1 = [bins[tag]['min'], *sorted(Ref1), bins[tag]['max']]
            y_Ref1 = np.linspace(0, 1, len(x_Ref1))
            (iv, ii) = np.unique(x_Ref1, return_index=True)
            i_CDF_1 = np.interp(bin_centers, iv, y_Ref1[ii])

            x_Ref2 = [bins[tag]['min'], *sorted(Ref2), bins[tag]['max']]
            y_Ref2 = np.linspace(0, 1, len(x_Ref2))
            (iv, ii) = np.unique(x_Ref2, return_index=True)
            i_CDF_2 = np.interp(bin_centers, iv, y_Ref2[ii])

            # EMDs computed with interpolated CDFs
            i_EMD_21 = sum(abs(i_CDF_2-i_CDF_1)) * bins[tag]['width'] / max_emd

            kwargs['max_emd'] = max_emd
            kwargs['i_CDF_1'] = i_CDF_1
            kwargs['i_CDF_2'] = i_CDF_2
            kwargs['i_EMD_21'] = i_EMD_21

        if run_means:
            means_Ref1 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            Ref1_mean = Ref1.mean()
            Ref2_mean = Ref2.mean()
            kwargs['Ref1_mean'] = Ref1_mean
            kwargs['Ref2_mean'] = Ref2_mean

        if run_excess:
            excess_Ref1 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            population_median = medians[tag]
            kwargs['population_median'] = population_median

        # Spawn threads
        with Parallel(n_jobs=nprocs) as parallel:
            for s, sample_size in enumerate(sample_sizes):
                for p, prop_Ref1 in enumerate(proportions):

                    # Parallelise over bootstraps
                    results = parallel(delayed(estimate_Ref1)(sample_size, prop_Ref1, b, **kwargs)
                                       for b in range(bootstraps))

                    for b in range(bootstraps):
                        if run_means:
                            means_Ref1[s, p, b] = results[b]['means']
                        if run_excess:
                            excess_Ref1[s, p, b] = results[b]['excess']
                        if run_KDE:
                            KDE_fits[s, p, b] = results[b]['KDE']
                        if run_EMD:
                            mat_EMD_31[s, p, b] = results[b]['EMD_31']
                            mat_EMD_32[s, p, b] = results[b]['EMD_32']

                    if (it >= bar_element*iterations/max_bars):
                        sys.stdout.write('*')
                        sys.stdout.flush()
                        bar_element += 1
                    if (it >= iterations-1):
                        sys.stdout.write('|\n')
                        sys.stdout.flush()
                    it += 1

        elapsed = time.time() - t
        print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

        # Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
        if run_EMD:
            norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
            norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21
            if check_EMD:
                norm_EMD_dev = emd_dev_from_fit * bin_width / max_emd / i_EMD_21
                median_error = 100 * np.median(norm_EMD_dev, axis=2)  # Percentage

        if run_means:
            np.save('{}/means_{}'.format(out_dir, tag), means_Ref1)
        if run_excess:
            np.save('{}/excess_{}'.format(out_dir, tag), excess_Ref1)
        if run_KDE:
            np.save('{}/kde_{}'.format(out_dir, tag), KDE_fits)
        if run_EMD:
            np.save('{}/emd_31_{}'.format(out_dir, tag), norm_mat_EMD_31)
            np.save('{}/emd_32_{}'.format(out_dir, tag), norm_mat_EMD_32)
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    # run_bootstrap('T1GRS', scores, medians, bins, sample_sizes, proportions, bootstraps)
    # run_bootstrap('T2GRS', scores, medians, bins, sample_sizes, proportions, bootstraps)
