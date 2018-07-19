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
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import lmfit

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
proportions = np.arange(0.0, 1.01, 0.01)  # T1 propoertions

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
    def kde_T1(x, amp_T1):
        return amp_T1 * np.exp(kdes['Type 1'].score_samples(x[:, np.newaxis]))

    def kde_T2(x, amp_T2):
        return amp_T2 * np.exp(kdes['Type 2'].score_samples(x[:, np.newaxis]))

    model = lmfit.Model(kde_T1) + lmfit.Model(kde_T2)

    # @mem.cache
    # def fit_KDE(RM, model, params_mix, kernel, bins):
    #     x_KDE = np.linspace(bins[tag]['min'], bins[tag]['max'], len(RM)+2)
    #     #x_KDE = np.array([0.095, *np.sort(RM), 0.35])
    #     mix_kde = KernelDensity(kernel=kernel, bandwidth=bins[tag]['width']).fit(RM[:, np.newaxis])
    #     res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
    #     amp_T1 = res_mix.params['amp_T1'].value
    #     amp_T2 = res_mix.params['amp_T2'].value
    #     return amp_T1/(amp_T1+amp_T2)


def estimate_T1D(sample_size, prop_T1, b, **kwargs):

    nT1 = int(round(sample_size * prop_T1))
    nT2 = sample_size - nT1

    # Random sample from T1
    R1 = np.random.choice(T1, nT1, replace=True)

    # Random sample from T2
    R2 = np.random.choice(T2, nT2, replace=True)

    # Bootstrap mixture
    RM = np.concatenate((R1, R2))
    # xRM = np.linspace(0, 1, num=len(RM), endpoint=True)

    # assert sample_size == len(RM)

    # x = np.array([0.095, *np.sort(RM), 0.35])
    results = {}

    # ---------------------- Difference of Means method ----------------------
    if run_means:
        proportion_of_T1 = (RM.mean()-T2_mean)/(T1_mean-T2_mean)
        results['means'] = abs(proportion_of_T1)

    # -------------------------- Subtraction method --------------------------
    if run_excess:
        number_low = len(RM[RM <= population_median])
        number_high = len(RM[RM > population_median])
        proportion_T1 = (number_high - number_low)/sample_size
        results['excess'] = proportion_T1

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

    metric = "T1GRS"
    scores, bins, means, medians, prop_Ref1 = load_diabetes_data(metric)
    scores, bins, means, medians, prop_Ref1 = load_renal_data()

    for tag in metrics:

        print("Running bootstrap with {} scores...".format(tag))

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

        T1 = scores[tag]['T1']
        T2 = scores[tag]['T2']

        kwargs = {}
        kwargs[tag] = tag
        kwargs['T1'] = T1
        kwargs['T2'] = T2

        bin_width = bins[tag]['width']
        bin_centers = bins[tag]['centers']
        bin_edges = bins[tag]['edges']

        if run_KDE:
            bw = bin_width  # Bandwidth
            kdes = {}
            labels = ['Type 1', 'Type 2']  # Reference populations

            # Fit reference populations
            for data, label in zip([T1, T2], labels):
                kdes[label] = {}
                X = data[:, np.newaxis]
                # for kernel in ['gaussian', 'tophat', 'epanechnikov']:
                kde = KernelDensity(kernel=kernel, bandwidth=bin_width).fit(X)
                # kdes[label][kernel] = kde
                kdes[label] = kde

            # kernel = 'gaussian'  #' epanechnikov'

            params_mix = model.make_params()
            params_mix['amp_T1'].value = 1
            params_mix['amp_T2'].value = 1

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
            x_T1 = [bins[tag]['min'], *sorted(T1), bins[tag]['max']]
            y_T1 = np.linspace(0, 1, len(x_T1))
            (iv, ii) = np.unique(x_T1, return_index=True)
            i_CDF_1 = np.interp(bin_centers, iv, y_T1[ii])

            x_T2 = [bins[tag]['min'], *sorted(T2), bins[tag]['max']]
            y_T2 = np.linspace(0, 1, len(x_T2))
            (iv, ii) = np.unique(x_T2, return_index=True)
            i_CDF_2 = np.interp(bin_centers, iv, y_T2[ii])

            # EMDs computed with interpolated CDFs
            i_EMD_21 = sum(abs(i_CDF_2-i_CDF_1)) * bins[tag]['width'] / max_emd

            kwargs['max_emd'] = max_emd
            kwargs['i_CDF_1'] = i_CDF_1
            kwargs['i_CDF_2'] = i_CDF_2
            kwargs['i_EMD_21'] = i_EMD_21

        if run_means:
            means_T1D = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            T1_mean = T1.mean()
            T2_mean = T2.mean()
            kwargs['T1_mean'] = T1_mean
            kwargs['T2_mean'] = T2_mean

        if run_excess:
            excess_T1D = np.zeros((len(sample_sizes), len(proportions), bootstraps))
            population_median = medians[tag]
            kwargs['population_median'] = population_median

        # Spawn threads
        with Parallel(n_jobs=nprocs) as parallel:
            for s, sample_size in enumerate(sample_sizes):
                for p, prop_T1 in enumerate(proportions):

                    # Parallelise over bootstraps
                    results = parallel(delayed(estimate_T1D)(sample_size, prop_T1, b, **kwargs)
                                       for b in range(bootstraps))

                    for b in range(bootstraps):
                        if run_means:
                            means_T1D[s, p, b] = results[b]['means']
                        if run_excess:
                            excess_T1D[s, p, b] = results[b]['excess']
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
            np.save('{}/means_{}'.format(out_dir, tag), means_T1D)
        if run_excess:
            np.save('{}/excess_{}'.format(out_dir, tag), excess_T1D)
        if run_KDE:
            np.save('{}/kde_{}'.format(out_dir, tag), KDE_fits)
        if run_EMD:
            np.save('{}/emd_31_{}'.format(out_dir, tag), norm_mat_EMD_31)
            np.save('{}/emd_32_{}'.format(out_dir, tag), norm_mat_EMD_32)
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    # run_bootstrap('T1GRS', scores, medians, bins, sample_sizes, proportions, bootstraps)
    # run_bootstrap('T2GRS', scores, medians, bins, sample_sizes, proportions, bootstraps)
