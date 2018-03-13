#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

@author: ben
"""

import time
import math
import sys
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import lmfit

from joblib import Parallel, delayed, cpu_count
from joblib import Memory
# mem = Memory(cachedir='/tmp')

# if __name__ == '__main__':
nprocs = multiprocessing.cpu_count()
print('Running with {}:{} processors...'.format(nprocs, cpu_count()))

# Set random seed
np.random.seed(42)
# rng = np.random.RandomState(42)

np.seterr(divide='ignore', invalid='ignore')

# xls = pd.ExcelFile("data.xls")
# data = xls.parse()
data = pd.read_csv('data.csv', usecols=[1, 2])
data.rename(columns={'diabetes_type': 'type', 't1GRS': 'T1GRS'}, inplace=True)
data.describe()

# Arrays of T1GRS scores for each group
T1 = data.loc[data['type'] == 1, 'T1GRS'].as_matrix()
T2 = data.loc[data['type'] == 2, 'T1GRS'].as_matrix()
Mix = data.loc[data['type'] == 3, 'T1GRS'].as_matrix()
scores = {'T1': T1, 'T2': T2, 'Mix': Mix}  # Raw T1GRS scores
T1_mean = T1.mean()
T2_mean = T2.mean()

# ----------------------------- Bin the data ---------------------------------
N = data.count()[0]
bin_width = 0.005
bin_edges = np.arange(0.095, 0.35+bin_width, bin_width)
#bin_centers = np.arange(0.095+bin_width/2, 0.35+bin_width/2, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centres


verbose = False
plot_results = False
run_means = True
run_excess = True
run_KDE = True
run_EMD = True
check_EMD = False

if plot_results:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    mpl.style.use('seaborn')
    mpl.rc('figure', figsize=(12, 10))

if run_excess:
    # Excess method median of T1GRS from the whole population in Biobank
    population_median = 0.23137931525707245
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
    EMD_21 = sum(abs(np.cumsum(hc2/sum(hc2)) - np.cumsum(hc1/sum(hc1)))) * bin_width * max_emd
    EMD_31 = sum(abs(np.cumsum(hc3/sum(hc3)) - np.cumsum(hc1/sum(hc1)))) * bin_width * max_emd
    EMD_32 = sum(abs(np.cumsum(hc3/sum(hc3)) - np.cumsum(hc2/sum(hc2)))) * bin_width * max_emd

    # Interpolate the cdfs at the same points for comparison
    x_T1 = [0.095, *sorted(T1), 0.35]
    y_T1 = np.linspace(0, 1, len(x_T1))
    (iv, ii) = np.unique(x_T1, return_index=True)
    i_CDF_1 = np.interp(bin_centers, iv, y_T1[ii])

    x_T2 = [0.095, *sorted(T2), 0.35]
    y_T2 = np.linspace(0, 1, len(x_T2))
    (iv, ii) = np.unique(x_T2, return_index=True)
    i_CDF_2 = np.interp(bin_centers, iv, y_T2[ii])

    x_Mix = [0.095, *sorted(Mix), 0.35]
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

    if plot_results and False:
        fig, axes = plt.subplots(3, 1, sharex=True) #, squeeze=False)
        X_plot = np.linspace(0.1, 0.35, 1000)[:, np.newaxis]

        for data, label, ax in zip([T1, T2, Mix], labels, axes):

            kdes[label] = {}
            X = data[:, np.newaxis]

            # [‘gaussian’, ’tophat’, ’epanechnikov’, ’exponential’, ’linear’, ’cosine’]
            for kernel in ['gaussian', 'tophat', 'epanechnikov']:
                kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
                log_dens = kde.score_samples(X_plot)
                ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                        label="kernel = '{0}'; bandwidth = {1}".format(kernel, bw))
                kdes[label][kernel] = kde  #np.exp(log_dens)

            #ax.text(6, 0.38, "N={0} points".format(N))

            ax.legend(loc='upper left')
            ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
            ax.set_ylabel(label)

            #ax.set_xlim(-4, 9)
            #ax.set_ylim(-0.02, 0.4)
            #plt.show()

    #sp.interpolate.interp1d(X, y, X_new, y_new)

    for data, label in zip([T1, T2, Mix], labels):
        kdes[label] = {}
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov']:
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
            kdes[label][kernel] = kde

    kernel = 'gaussian'  #'epanechnikov'

    # TODO: Rethink
    n_bins = int(np.floor(np.sqrt(N)))
    (freqs_T1, bins) = np.histogram(T1, bins=n_bins)
    (freqs_T2, bins) = np.histogram(T2, bins=n_bins)
    (freqs_Mix, bins) = np.histogram(Mix, bins=n_bins)

    x = np.array((bins[:-1] + bins[1:])) / 2  # Bin centres
    y = Mix

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

    if plot_results:
        plt.figure()
        fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False)

        axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=labels[:-1])
        legend = axP.legend(facecolor='grey')
        #legend.get_frame().set_facecolor('grey')
        axP.set_title('Proportions of Type 1 and Type 2 vs T1GRS')

        plt.sca(axM)
        res_mix.plot_fit()

        axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")

        plt.sca(axR)
        res_mix.plot_residuals()

        #plt.sca(axI)
        axI.plot(x, kde1, label='Type 1')
        axI.plot(x, kde2, label='Type 2')

    if verbose:
        print(res_mix.fit_report())
        print('T2/T1 =', amp_T2/amp_T1)
        print('')
        print('\nParameter confidence intervals:')
        print(res_mix.ci_report())  # --> res_mix.ci_out # See also res_mix.conf_interval()

# ------------ Summarise proportions for the whole distribution --------------
if run_means:
    print('Proportions based on means')
    print('% of Type 1:', (Mix.mean()-T2.mean())/(T1.mean()-T2.mean()))
    print('% of Type 2:', (1-(Mix.mean()-T2.mean())/(T1.mean()-T2.mean())))

if run_excess:
    print('Proportions based on excess')
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


bootstraps = 100
sample_sizes = np.array(range(100, 2000, 200)) #np.array(range(100, 3100, 100))
proportions = np.arange(0.01, 1.01, 0.02)

if run_KDE:
    #KDE_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))
    #KDE_rms_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))
    KDE_fits = np.zeros((len(sample_sizes), len(proportions), bootstraps))

    #@mem.cache
    def fit_KDE(RM, model, params_mix, kernel, bw):
        x_KDE = np.linspace(0.095, 0.35, len(RM)+2)
        #x_KDE = np.array([0.095, *np.sort(RM), 0.35])
        mix_kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(RM[:, np.newaxis])
        res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
        amp_T1 = res_mix.params['amp_T1'].value
        amp_T2 = res_mix.params['amp_T2'].value
        #KDE_fits[s, p, b] = amp_T1/(amp_T1+amp_T2)
        return amp_T1/(amp_T1+amp_T2)

if run_EMD:
    mat_EMD_31 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
    mat_EMD_32 = np.zeros((len(sample_sizes), len(proportions), bootstraps))
    if check_EMD:
        emd_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))
        rms_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), bootstraps))

if run_means:
    means_T1D = np.zeros((len(sample_sizes), len(proportions), bootstraps))

if run_excess:
    excess_T1D = np.zeros((len(sample_sizes), len(proportions), bootstraps))


def estimate_T1D(sample_size, prop_T1, b):

    nT1 = int(round(sample_size * prop_T1))
    nT2 = sample_size - nT1

    # Random sample from T1
    R1 = np.random.choice(T1, nT1, replace=True)

    # Random sample from T2
    R2 = np.random.choice(T2, nT2, replace=True)

    # Bootstrap mixture
    RM = np.concatenate((R1, R2))
    #xRM = np.linspace(0, 1, num=len(RM), endpoint=True)

    #x = np.array([0.095, *np.sort(RM), 0.35])
    results = {}

    # ---------------------- Difference of Means method ----------------------
    if run_means:
        proportion_of_T1 = 100*((RM.mean()-T2_mean)/(T1_mean-T2_mean))
        #means_T1D[s, p, b] = abs(proportion_of_T1)
        results['means'] = abs(proportion_of_T1)


    # -------------------------- Subtraction method --------------------------
    if run_excess:
        number_low = len(RM[RM <= population_median])
        number_high = len(RM[RM > population_median])
        high = number_high - number_low
        low = 2*number_low
        proportion_T1 = 100*(high/(low+high))
        #excess_T1D[s, p, b] = proportion_T1
        results['excess'] = proportion_T1


    # ------------------------------ KDE method ------------------------------
    if run_KDE:
        #KDE_fits[s, p, b] = fit_KDE(RM, model, params_mix, kernel, bw)
        results['KDE'] = fit_KDE(RM, model, params_mix, kernel, bw)


    # ------------------------------ EMD method ------------------------------
    if run_EMD:
        # Interpolated cdf (to compute EMD)
        x = [0.095, *np.sort(RM), 0.35]
        y = np.linspace(0, 1, num=len(x), endpoint=True)
        (iv, ii) = np.unique(x, return_index=True)
        si_CDF_3 = np.interp(bin_centers, iv, y[ii])

        # Compute EMDs
        i_EMD_31 = sum(abs(si_CDF_3-i_CDF_1)) * bin_width / max_emd
        i_EMD_32 = sum(abs(si_CDF_3-i_CDF_2)) * bin_width / max_emd
        #mat_EMD_31[s, p, b] = i_EMD_31  # emds to compute proportions
        #mat_EMD_32[s, p, b] = i_EMD_32  # emds to compute proportions

        if check_EMD:
            # These were computed to check that the EMD computed proportions fit the mixture's CDF
            EMD_diff = si_CDF_3 - ((1-i_EMD_31/i_EMD_21)*i_CDF_1 + (1-i_EMD_32/i_EMD_21)*i_CDF_2)
            emd_dev_from_fit[s, p, b] = sum(EMD_diff)  # deviations from fit measured with emd
            rms_dev_from_fit[s, p, b] = math.sqrt(sum(EMD_diff**2)) / len(si_CDF_3)  # deviations from fit measured with rms

        results['EMD_31'] = i_EMD_31
        results['EMD_32'] = i_EMD_32

    return results


# Setup progress bar
iterations = len(sample_sizes) * len(proportions) # * bootstraps  #KDE_fits.size
max_bars = 78    # number of dots in progress bar
if iterations < max_bars:
    max_bars = iterations   # if less than 20 points in scan, shorten bar
print("|" + max_bars*"-" + "|     Bootstrap progress")
sys.stdout.write('|')
sys.stdout.flush()  # print start of progress bar


t = time.time()  # Start timer
it = 0
bar_element = 0
#for b in range(bootstraps):
with Parallel(n_jobs=nprocs) as parallel:
    for s, sample_size in enumerate(sample_sizes):
        for p, prop_T1 in enumerate(proportions):

            # Parallelise over bootstraps
            results = parallel(delayed(estimate_T1D)(sample_size, prop_T1, b)
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
                sys.stdout.write('|     done  \n')
                sys.stdout.flush()
            it += 1


elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds'.format(elapsed))


# Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
if run_EMD:
    norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
    norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21
    if check_EMD:
        norm_EMD_dev = emd_dev_from_fit * bin_width / max_emd / i_EMD_21
        median_error = 100 * np.median(norm_EMD_dev, axis=2)  # Percentage

if run_means:
    np.save('means', means_T1D)
if run_excess:
    np.save('excess', excess_T1D)
if run_KDE:
    np.save('kde', KDE_fits)
if run_EMD:
    np.save('emd_31', norm_mat_EMD_31)
    np.save('emd_32', norm_mat_EMD_32)
np.save('sample_sizes', sample_sizes)
np.save('proportions', proportions)

# ------------------------------- Plot results -------------------------------

if plot_results:
    proportions_rev = proportions[::-1]
    plot_relative_error = False
    plot_absolute_error = True
    plot_standard_deviation = True

    if plot_relative_error:
        #TODO: Plot SD around estimated proportion
        plt.figure()
        #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
        #plt.colorbar()

        levels = np.array([5.0]) #np.array([0.1, 1.0])  # Percentage relative error

        if run_EMD:
            relative_error_EMD_T1 = 100*(np.median(1-norm_mat_EMD_31, axis=2)-proportions)/proportions
            relative_error_EMD_T2 = 100*(np.median(1-norm_mat_EMD_32, axis=2)-proportions_rev)/proportions_rev
            max_relative_error_EMD = np.maximum(relative_error_EMD_T1, relative_error_EMD_T2)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_relative_error_EMD),
                             levels, colors='r')

        if run_means:
            #relative_error_means = 100*(np.median(means_T1D/100, axis=2)-proportions)/proportions
            relative_error_means_T1 = 100*(np.median(means_T1D/100, axis=2)-proportions)/proportions
            relative_error_means_T2 = 100*(np.median(1-means_T1D/100, axis=2)-proportions_rev)/proportions_rev
            max_relative_error_means = np.maximum(relative_error_means_T1, relative_error_means_T2)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_relative_error_means),
                             levels, colors='k')

        if run_excess:
            # adjusted for fact underestimates by 8%
            relative_error_excess_T1_adj = 100*(np.median((excess_T1D/0.92)/100, axis=2)-proportions)/proportions
            relative_error_excess_T2_adj = 100*(np.median(1-(excess_T1D/0.92)/100, axis=2)-proportions_rev)/proportions_rev
            max_relative_error_excess_adj = np.maximum(relative_error_excess_T1_adj, relative_error_excess_T2_adj)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_relative_error_excess_adj),
                             levels, colors='b')

    if plot_absolute_error:
        plt.figure()
        #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
        #plt.colorbar()

        levels = np.array([(0.05)]) #np.array([0.1, 1.0])  # # Percentage relative error

        if run_EMD:
            relative_error_EMD_T1D = np.abs(((np.mean(1-norm_mat_EMD_31, axis=2))/proportions)-1)
            relative_error_EMD_T2D = np.abs(((1-(np.mean(1-norm_mat_EMD_31, axis=2)))/proportions_rev)-1)
            max_relative_error_emd_abs = np.maximum(relative_error_EMD_T1D, relative_error_EMD_T2D)
            CS = plt.contour(proportions, sample_sizes, (max_relative_error_emd_abs),
                             levels, colors='r')

        if run_means:
            relative_error_means_T1D = np.abs(((np.mean(means_T1D/100, axis=2))/proportions)-1)
            relative_error_means_T2D = np.abs(((1-(np.mean(means_T1D/100, axis=2)))/proportions_rev)-1)
            max_relative_error_means_abs = np.maximum(relative_error_means_T1D, relative_error_means_T2D)
            CS = plt.contour(proportions, sample_sizes, (max_relative_error_means_abs),
                             levels, colors='k')

        if run_excess:
            excess_T1D = (excess_T1D/0.92) #adjustment for missing 8%
            relative_error_excess_T1D = np.abs(((np.mean(excess_T1D/100, axis=2))/proportions)-1)
            relative_error_excess_T2D = np.abs(((1-(np.mean(excess_T1D/100, axis=2)))/proportions_rev)-1)
            max_relative_error_excess_abs = np.maximum(relative_error_excess_T1D, relative_error_excess_T2D)
            CS = plt.contour(proportions, sample_sizes, (max_relative_error_excess_abs),
                             levels, colors='g')

    if plot_standard_deviation:
        plt.figure()
        #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
        #plt.colorbar()

        levels = np.array([3.0]) #np.array([0.1, 1.0])  # # Percentage relative error

        if run_EMD:
            adjusted_EMD = 100*(1-norm_mat_EMD_31)
            dev_EMD_t1 = np.std(adjusted_EMD, axis=2)
            dev_EMD_t2 = np.std(100-adjusted_EMD, axis=2)
            max_dev_EMD = np.maximum(dev_EMD_t1, dev_EMD_t2)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_dev_EMD),
                             levels, colors='r')

        if run_means:
            dev_means_t1 = np.std(means_T1D, axis=2)
            dev_means_t2 = np.std(100-means_T1D, axis=2)
            max_dev_means = np.maximum(dev_means_t1, dev_means_t2)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_dev_means),
                             levels, colors='k')

        if run_excess:
            excess_T1D_adj = (excess_T1D) #adjustment for missing 8%
            dev_excess_t1 = np.std(excess_T1D_adj, axis=2)
            dev_excess_t2 = np.std(100-excess_T1D_adj, axis=2)
            max_dev_excess = np.maximum(dev_excess_t1, dev_excess_t2)
            CS = plt.contour(proportions, sample_sizes, np.abs(max_dev_excess),
                             levels, colors='g')

    if verbose:
        plt.figure()
        fig, (axEMD, axMeans, axExcess) = plt.subplots(3, 1, sharex=True, sharey=False)
        axEMD.contourf(proportions, sample_sizes, relative_error_EMD, cmap='bwr')
        #plt.colorbar()
        #lim = np.amax(abs(relative_error_EMD))
        #plt.clim(-lim, lim)
        axMeans.contourf(proportions, sample_sizes, relative_error_means, cmap='bwr')
        #plt.colorbar()
        #lim = np.amax(abs(relative_error_Means))
        #plt.clim(-lim, lim)
        axExcess.contourf(proportions, sample_sizes, relative_error_excess, cmap='bwr')
        #fig.colorbar()
        #lim = np.amax(abs(relative_error_Excess))
        #plt.clim(-lim, lim)

    if verbose:
        # Deviation from fit
        plt.figure()
        plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
        plt.colorbar()

        #contour3(median(emd_dev_from_fit/0.0128,3),[0.01 0.001]*max(emd_dev_from_fit(:)/0.0128),'r','LineWidth',3)
        levels = np.array([0.1, 1.0]) * np.amax(norm_EMD_dev)  # Percentage
        CS = plt.contour(proportions, sample_sizes, np.amax(norm_EMD_dev, axis=2), levels)
        plt.clabel(CS, inline=1, fontsize=10)

        plt.xlabel('Proportion (Type 1)')
        plt.ylabel('Sample size')
        plt.title('Median propotion error from true proportion (as a % of maximum EMD error)\nContours represent maximum error')

        # Error T1
        plt.figure()
        rel_err_31 = 100*(np.median((1-norm_mat_EMD_31), axis=2) - proportions)/proportions
        plt.contourf(proportions, sample_sizes, rel_err_31, cmap='bwr')
        plt.colorbar()
        lim = np.amax(abs(rel_err_31))
        plt.clim(-lim, lim)

        # 1 & 5% relative error contour the other proportion
        CS = plt.contour(proportions, sample_sizes, rel_err_31, [1, 5])
        plt.clabel(CS, inline=1, fontsize=10)

        plt.xlabel('Proportion (Type 1)')
        plt.ylabel('Sample size')
        plt.title('Relative % error from Type 1 population')

        # Error T2
        plt.figure()
        proportions_rev = proportions[::-1]
        rel_err_32 = 100*(np.median((1-norm_mat_EMD_32), axis=2) - proportions_rev)/proportions_rev
        plt.contourf(proportions_rev, sample_sizes, rel_err_32, cmap='bwr')
        plt.colorbar()
        lim = np.amax(abs(rel_err_32))
        plt.clim(-lim, lim)

        # 1 & 5% relative error contour the other proportion
        CS = plt.contour(proportions_rev, sample_sizes, rel_err_32, [1, 5])
        plt.clabel(CS, inline=1, fontsize=10)

        plt.xlim(0.99, 0.01)  # Reverse axis
        plt.xlabel('Proportion (Type 2)')
        plt.ylabel('Sample size')
        plt.title('Relative % error from Type 2 population')

        # Max Error
        ers = np.zeros((len(sample_sizes), len(proportions), 2))
        ers[:, :, 0] = 100*(np.median(1-mat_EMD_31/i_EMD_21, axis=2) - proportions_rev)/proportions_rev  # N.B. Swapped indicies
        ers[:, :, 1] = 100*(np.median(1-mat_EMD_32/i_EMD_21, axis=2) - proportions)/proportions
        max_ers = np.amax(abs(ers), axis=2)

        plt.figure()
        plt.contourf(proportions, sample_sizes, max_ers, cmap='viridis_r')
        plt.colorbar()

        # 1 & 5% relative error contour the max error proportion
        CS = plt.contour(proportions, sample_sizes, max_ers, [1, 5])
        plt.clabel(CS, inline=1, fontsize=10)

        plt.xlabel('Proportion (Type 1)')
        plt.ylabel('Sample size')
        plt.title('Maximum % relative error from either population')
