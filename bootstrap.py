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


def analyse_mixture(tag, scores, means, medians, bins):

    T1 = scores[tag]['T1']
    T2 = scores[tag]['T2']
    Mix = scores[tag]['Mix']
    population_median = medians[tag]
    bin_width = bins[tag]['width']
    bin_centers = bins[tag]['centers']
    bin_edges = bins[tag]['edges']

    print('Runnings {} mixture analysis...'.format(tag))
    print('--------------------------------------------------------------------------------')

    if run_excess:
        # Excess method median of GRS from the whole population in Biobank
        high = len(Mix[Mix > population_median]) - len(Mix[Mix <= population_median])
        low = 2*len(Mix[Mix <= population_median])

    # -------------------------------- EMD method --------------------------------

    (hc1, _) = np.histogram(T1, bins=bin_edges)
    (hc2, _) = np.histogram(T2, bins=bin_edges)
    (hc3, _) = np.histogram(Mix, bins=bin_edges)
    counts = {'T1': hc1, 'T2': hc2, 'Mix': hc3}  # Binned score frequencies

    if run_EMD:
        # EMDs computed with histograms (compute pair-wise EMDs between the 3 histograms)
        max_emd = bin_edges[-1] - bin_edges[0]
        EMD_21 = sum(abs(np.cumsum(hc2/sum(hc2)) - np.cumsum(hc1/sum(hc1)))) * bin_width / max_emd
        EMD_31 = sum(abs(np.cumsum(hc3/sum(hc3)) - np.cumsum(hc1/sum(hc1)))) * bin_width / max_emd
        EMD_32 = sum(abs(np.cumsum(hc3/sum(hc3)) - np.cumsum(hc2/sum(hc2)))) * bin_width / max_emd

        # Interpolate the cdfs at the same points for comparison
        x_T1 = [bins[tag]['min'], *sorted(T1), bins[tag]['max']]
        y_T1 = np.linspace(0, 1, len(x_T1))
        (iv, ii) = np.unique(x_T1, return_index=True)
        i_CDF_1 = np.interp(bin_centers, iv, y_T1[ii])

        x_T2 = [bins[tag]['min'], *sorted(T2), bins[tag]['max']]
        y_T2 = np.linspace(0, 1, len(x_T2))
        (iv, ii) = np.unique(x_T2, return_index=True)
        i_CDF_2 = np.interp(bin_centers, iv, y_T2[ii])

        x_Mix = [bins[tag]['min'], *sorted(Mix), bins[tag]['max']]
        y_Mix = np.linspace(0, 1, len(x_Mix))
        (iv, ii) = np.unique(x_Mix, return_index=True)
        i_CDF_3 = np.interp(bin_centers, iv, y_Mix[ii])

        # EMDs computed with interpolated CDFs
        i_EMD_21 = sum(abs(i_CDF_2-i_CDF_1)) * bin_width / max_emd
        i_EMD_31 = sum(abs(i_CDF_3-i_CDF_1)) * bin_width / max_emd
        i_EMD_32 = sum(abs(i_CDF_3-i_CDF_2)) * bin_width / max_emd

    if run_KDE:
        # ------------------------------ KDE method ------------------------------

        bw = bin_width  # Bandwidth

        kdes = {}
        labels = ['Type 1', 'Type 2', 'Mixture']

#        # TODO: Rethink
#        n_bins = int(np.floor(np.sqrt(N)))
#        (freqs_T1, bins) = np.histogram(T1, bins=n_bins)
#        (freqs_T2, bins) = np.histogram(T2, bins=n_bins)
#        (freqs_Mix, bins) = np.histogram(Mix, bins=n_bins)
#
#        x = np.array((bins[:-1] + bins[1:])) / 2  # Bin centres
#        y = Mix

        (freqs_Mix, _) = np.histogram(Mix, bins=bins[tag]['edges'])
        x = bins[tag]['centers']

        # sp.interpolate.interp1d(X, y, X_new, y_new)

        for data, label in zip([T1, T2, Mix], labels):
            kdes[label] = {}
            X = data[:, np.newaxis]
            for kernel in ['gaussian', 'tophat', 'epanechnikov']:
                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
                kdes[label][kernel] = kde

        kernel = 'gaussian'  #'epanechnikov'

        # TODO: Refactor
        # Define the KDE models
        def kde_T1(x, amp_T1):
            return amp_T1 * np.exp(kdes['Type 1'][kernel].score_samples(x[:, np.newaxis]))

        def kde_T2(x, amp_T2):
            return amp_T2 * np.exp(kdes['Type 2'][kernel].score_samples(x[:, np.newaxis]))

        model_T1 = lmfit.Model(kde_T1)
        model_T2 = lmfit.Model(kde_T2)

        model = model_T1 + model_T2
        params_mix = model.make_params()
        params_mix['amp_T1'].value = 1
        params_mix['amp_T2'].value = 1

        res_mix = model.fit(freqs_Mix, x=x, params=params_mix)

        dely = res_mix.eval_uncertainty(sigma=3)

        amp_T1 = res_mix.params['amp_T1'].value
        amp_T2 = res_mix.params['amp_T2'].value

        kde1 = kde_T1(x, amp_T1)
        kde2 = kde_T2(x, amp_T2)

        if verbose:
            print(res_mix.fit_report())
            print('T2/T1 =', amp_T2/amp_T1)
            print('')
            print('\nParameter confidence intervals:')
            print(res_mix.ci_report())  # --> res_mix.ci_out # See also res_mix.conf_interval()

    # ------------ Summarise proportions for the whole distribution --------------
    if run_means:
        print('Proportions based on means')
        print('% of Type 1:', (Mix.mean()-means[tag]['T2'])/(means[tag]['T1']-means[tag]['T2']))
        print('% of Type 2:', (1-(Mix.mean()-means[tag]['T2'])/(means[tag]['T1']-means[tag]['T2'])))

    if run_excess:
        print('Proportions based on excess')
        # TODO: Flip these around for when using the T2GRS
        print('% of Type 1:', (high/(low+high)))
        print('% of Type 2:', (1-(high/(low+high))))

    print('Proportions based on counts')
    print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
    print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))

    if run_EMD:
        print("Proportions based on Earth Mover's Distance (histogram values):")
        print('% of Type 1:', 1-EMD_31/EMD_21)
        print('% of Type 2:', 1-EMD_32/EMD_21)

        print("Proportions based on Earth Mover's Distance (interpolated values):")
        print('% of Type 1:', 1-i_EMD_31/i_EMD_21)
        print('% of Type 2:', 1-i_EMD_32/i_EMD_21)

    if run_KDE:
        print('Proportions based on KDEs')
        print('% of Type 1:', amp_T1/(amp_T1+amp_T2))
        print('% of Type 2:', amp_T2/(amp_T1+amp_T2))

    print('--------------------------------------------------------------------------------\n\n')


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

    # xls = pd.ExcelFile("data.xls")
    # data = xls.parse()
    # data = pd.read_csv('t1d_t2d_bootstrap_data.csv', usecols=[1, 2]) #t2 included data
    # data = pd.read_csv('data.csv', usecols=[1, 2])  #
    # data = pd.read_csv('data_biobank_all.csv', usecols=[0, 1, 2])
    data = pd.read_csv(dataset)
    # data = pd.read_csv('c_pep_defined2.csv')
    # data = pd.read_csv('data_bioban_t2_c_pep.csv')
    # data.rename(columns={'diabetes_type': 'type', 't1GRS': 'T1GRS'}, inplace=True)
    # data.rename(columns={'diabetes_type': 'type', 't2GRS': 'T1GRS'}, inplace=True) # looking at T2GRS
    data.rename(columns=headers, inplace=True)
    data.describe()

    scores = {}
    means = {}

    # Arrays of T1GRS scores for each group
    scores['T1GRS'] = {'T1': data.loc[data['type'] == 1, 'T1GRS'].as_matrix(),
                       'T2': data.loc[data['type'] == 2, 'T1GRS'].as_matrix(),
                       'Mix': data.loc[data['type'] == 3, 'T1GRS'].as_matrix()}

    means['T1GRS'] = {'T1': scores['T1GRS']['T1'].mean(),
                      'T2': scores['T1GRS']['T2'].mean()}

    # Arrays of T2GRS scores for each group
    scores['T2GRS'] = {'T1': data.loc[data['type'] == 1, 'T2GRS'].as_matrix(),
                       'T2': data.loc[data['type'] == 2, 'T2GRS'].as_matrix(),
                       'Mix': data.loc[data['type'] == 3, 'T2GRS'].as_matrix()}

    means['T2GRS'] = {'T1': scores['T2GRS']['T1'].mean(),
                      'T2': scores['T2GRS']['T2'].mean()}

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

    analyse_mixture('T1GRS', scores, means, medians, bins)
    analyse_mixture('T2GRS', scores, means, medians, bins)

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
