#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:04:57 2018

@author: ben
"""

import os
import time
from collections import OrderedDict
# import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

import analyse_mixture as pe
from datasets import (load_diabetes_data, load_renal_data)


def plot_bootstrap_results(out_dir, label, fig, axes):

    PROPORTIONS_T1D = np.load('{}/proportions_{}.npy'.format(out_dir, label))
    SAMPLE_SIZES = np.load('{}/sample_sizes_{}.npy'.format(out_dir, label))
    PROPORTIONS_T2D = PROPORTIONS_T1D[::-1]

    if os.path.isfile('{}/means_{}.npy'.format(out_dir, label)):
        MEANS_T1D = np.load('{}/means_{}.npy'.format(out_dir, label))
        MEANS_T2D = 1 - MEANS_T1D
        PLOT_MEANS = True
    if os.path.isfile('{}/excess_{}.npy'.format(out_dir, label)):
        EXCESS_T1D = np.load('{}/excess_{}.npy'.format(out_dir, label))
        EXCESS_T2D = 1 - EXCESS_T1D
        if ADJUST_EXCESS:
            EXCESS_T1D /= 0.92  # adjusted for fact underestimates by 8%
        PLOT_EXCESS = True
    if os.path.isfile('{}/emd_31_{}.npy'.format(out_dir, label)) and \
       os.path.isfile('{}/emd_32_{}.npy'.format(out_dir, label)):
        EMD_31 = np.load('{}/emd_31_{}.npy'.format(out_dir, label))
        EMD_32 = np.load('{}/emd_32_{}.npy'.format(out_dir, label))
        PLOT_EMD = True
    if os.path.isfile('{}/emd_{}.npy'.format(out_dir, label)):
        # HACK!
        # EMD_P1 = 1 - EMD_31
        EMD_32 = np.load('{}/emd_{}.npy'.format(out_dir, label))
        EMD_31 = 1 - EMD_32
        PLOT_EMD = True
    if os.path.isfile('{}/kde_{}.npy'.format(out_dir, label)):
        KDE_T1D = np.load('{}/kde_{}.npy'.format(out_dir, label))
        KDE_T2D = 1 - KDE_T1D
        PLOT_KDE = True


    error_MEANS_T1 = average(MEANS_T1D, axis=2) - PROPORTIONS_T1D
    error_MEANS_T2 = average(MEANS_T2D, axis=2) - PROPORTIONS_T2D
    error_EXCESS_T1 = average(EXCESS_T1D, axis=2) - PROPORTIONS_T1D
    error_EXCESS_T2 = average(EXCESS_T2D, axis=2) - PROPORTIONS_T2D
    error_EMD_T1 = average(1-EMD_31, axis=2) - PROPORTIONS_T1D
    error_EMD_T2 = average(1-EMD_32, axis=2) - PROPORTIONS_T2D
    error_KDE_T1 = average(KDE_T1D, axis=2) - PROPORTIONS_T1D
    error_KDE_T2 = average(KDE_T2D, axis=2) - PROPORTIONS_T2D

    if PLOT_RELATIVE_ERROR:

        LEVELS = np.array([5.0])  # Percentage relative error

        relative_error_MEANS_T1 = 100*error_MEANS_T1/PROPORTIONS_T1D
        relative_error_MEANS_T2 = 100*error_MEANS_T2/PROPORTIONS_T2D
        max_relative_error_MEANS = np.maximum(np.abs(relative_error_MEANS_T1),
                                              np.abs(relative_error_MEANS_T2))

        relative_error_EXCESS_T1 = 100*error_EXCESS_T1/PROPORTIONS_T1D
        relative_error_EXCESS_T2 = 100*error_EXCESS_T2/PROPORTIONS_T2D
        max_relative_error_EXCESS = np.maximum(np.abs(relative_error_EXCESS_T1),
                                               np.abs(relative_error_EXCESS_T2))

        relative_error_EMD_T1 = 100*error_EMD_T1/PROPORTIONS_T1D
        relative_error_EMD_T2 = 100*error_EMD_T2/PROPORTIONS_T2D
        max_relative_error_EMD = np.maximum(np.abs(relative_error_EMD_T1),
                                            np.abs(relative_error_EMD_T2))

        relative_error_KDE_T1 = 100*error_KDE_T1/PROPORTIONS_T1D
        relative_error_KDE_T2 = 100*error_KDE_T2/PROPORTIONS_T2D
        max_relative_error_KDE = np.maximum(np.abs(relative_error_KDE_T1),
                                            np.abs(relative_error_KDE_T2))

        datasets = [max_relative_error_MEANS, max_relative_error_EXCESS,
                    max_relative_error_EMD, max_relative_error_KDE]

        # Plot contours for each method on one figure
        plt.figure()
        # plt.title('{} : Maximum relative error (%)\nContours at {}'.format(metric, LEVELS))
        plt.xlabel('Proportion T1D')
        plt.ylabel('Sample size')

        for label, colour, data in zip(LABELS, COLOURS, datasets):
            CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            CS.collections[0].set_label(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/relative_{}.png'.format(label))

        # Plot shaded regions for each method on individual subplots
        if LINEAR_COLOURBAR:
            SHADING_LEVELS = np.arange(0, 100, 5)
        else:
            SHADING_LEVELS = np.logspace(-2, 2, num=25)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        plt.suptitle('{} : Maximum relative error (%) [Contours at {}]'.format(label, LEVELS))
        for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
            ax.set_title(label)
            CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                             SHADING_LEVELS, cmap='viridis_r',
                             locator=mpl.ticker.LogLocator())  # , extend='max'
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            fig.colorbar(CS, ax=ax)  # , norm=mpl.colors.LogNorm())
        plt.tight_layout()
        plt.savefig('figs/relative_sub_{}.png'.format(label))


    if PLOT_ABSOLUTE_ERROR:

        LEVELS = np.array([0.02])  # Percentage relative error

        max_abs_error_MEANS = np.maximum(np.abs(error_MEANS_T1),
                                         np.abs(error_MEANS_T2))

        max_abs_error_EXCESS = np.maximum(np.abs(error_EXCESS_T1),
                                          np.abs(error_EXCESS_T2))

        max_abs_error_EMD = np.maximum(np.abs(error_EMD_T1),
                                       np.abs(error_EMD_T2))

        max_abs_error_KDE = np.maximum(np.abs(error_KDE_T1),
                                       np.abs(error_KDE_T2))

        if ABSOLUTE_ERROR:
            datasets = [max_abs_error_MEANS, max_abs_error_EXCESS,
                        max_abs_error_EMD, max_abs_error_KDE]
        else:
            datasets = [error_MEANS_T1, error_EXCESS_T1,
                        error_EMD_T1, error_KDE_T1]

        if False:
            # Plot contours for each method on one figure
            plt.figure()
            plt.title('{} : Maximum absolute error\nContours at {}'.format(label, LEVELS))
            plt.xlabel('Proportion T1D')
            plt.ylabel('Sample size')

            for label, colour, data in zip(LABELS, COLOURS, datasets):
                CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
                CS.collections[0].set_label(label)
            plt.legend()
            plt.tight_layout()
            plt.savefig('figs/absolute_{}.png'.format(label))

        # Plot shaded regions for each method on individual subplots
        if LINEAR_COLOURBAR:
            if ABSOLUTE_ERROR:
                SHADING_LEVELS = np.arange(0, 0.051, 0.005)
            else:
                SHADING_LEVELS = np.arange(-0.05, 0.051, 0.005)
        else:
            SHADING_LEVELS = np.logspace(-4, -1, num=19)

        if ABSOLUTE_ERROR:
            cmap = 'viridis_r'
        else:
            #cmap = sns.color_palette("RdBu", n_colors=len(SHADING_LEVELS))
            cmap = "bwr"

        # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # plt.suptitle('{} : Maximum absolute error [Contours at {}]'.format(metric, LEVELS))
        colour = [sns.color_palette()[6]]
        # for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
        for ax, label, data in zip(axes.ravel(), LABELS, datasets):
            ax.set_title(label)
            if LINEAR_COLOURBAR:
                if ABSOLUTE_ERROR:
                    CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                                     SHADING_LEVELS, cmap=cmap, extend='max')
                else:
                    CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                                     SHADING_LEVELS, cmap=cmap, extend='both')  # , vmin=-.05, vmax=.05)
            else:
                CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                                 SHADING_LEVELS, cmap=cmap,
                                 locator=mpl.ticker.LogLocator())  # , extend='max'
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            #if label == "Means" or label == "EMD":
            fig.colorbar(CS, ax=ax)  #, norm=mpl.colors.LogNorm())
        # plt.tight_layout()
        plt.savefig('figs/absolute_sub_{}.png'.format(data_label))


    if PLOT_STANDARD_DEVIATION:

        LEVELS = np.array([0.05])

        dev_MEANS = deviation(MEANS_T1D, axis=2)

        dev_EXCESS = deviation(EXCESS_T1D, axis=2)

        dev_EMD_T1 = deviation(EMD_31, axis=2)
        dev_EMD_T2 = deviation(EMD_32, axis=2)
        max_dev_EMD = np.maximum(dev_EMD_T1, dev_EMD_T2)

        dev_KDE = deviation(KDE_T1D, axis=2)

        datasets = [dev_MEANS, dev_EXCESS, max_dev_EMD, dev_KDE]

        if False:
            # Plot shaded regions for each method on individual subplots
            plt.figure()
            # plt.title('{} : Maximum standard deviation\nContours at {}'.format(metric, LEVELS))
            plt.xlabel('Proportion T1D')
            plt.ylabel('Sample size')

            for label, colour, data in zip(LABELS, COLOURS, datasets):
                CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
                CS.collections[0].set_label(label)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('figs/deviation_{}.png'.format(label))

        # Plot shaded regions for each method on individual subplots
        SHADING_LEVELS = np.arange(0.005, 0.1, 0.005)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        # plt.suptitle('{} : Maximum standard deviation [Contours at {}]'.format(metric, LEVELS))
        # for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
        for ax, label, data in zip(axes.ravel(), LABELS, datasets):
            ax.set_title(label)
            CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                             SHADING_LEVELS, cmap='viridis_r', extend='both')
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            fig.colorbar(CS, ax=ax)
        plt.tight_layout()
        plt.savefig('figs/deviation_sub_{}.png'.format(label))



def plot_distributions(scores, bins, data_label, ax=None): #, kernel, bandwidth):

    if not ax:
        f, ax = plt.subplots()
    #sns.set_style("ticks")

    #with sns.axes_style("ticks"):
    sns.distplot(scores['Ref1'], bins=bins['edges'], norm_hist=False, label="Ref1: N={}".format(len(scores['Ref1'])), ax=ax)
    sns.distplot(scores['Ref2'], bins=bins['edges'], norm_hist=False, label="Ref2: N={}".format(len(scores['Ref2'])), ax=ax)
    sns.distplot(scores['Mix'], bins=bins['edges'], norm_hist=False, label="Mix: N={}".format(len(scores['Mix'])), ax=ax)

    sns.despine(top=True, bottom=False, left=True, right=False, trim=True)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
#    if len(indep_vars) > 1:
        # Use jointplot
    ax.legend()
    # plt.savefig('figs/distributions_{}.png'.format(data_label))


def plot_bootstraps(df_bs, prop_Ref1=None, ax=None, ylims=None):

    c = sns.color_palette()[-3]  # 'gray'

    if not ax:
        f, ax = plt.subplots()
    if ylims:
        ax.set_ylim(ylims)

    # Adding this because a bug means that moving yaxis to right removes tickmarks
    #sns.set_style("whitegrid")

    # Draw violin plots of bootstraps
    sns.violinplot(data=df_bs, ax=ax)  # , inner="stick")
    sns.despine(top=True, bottom=True, left=True, right=False, trim=True)

    if prop_Ref1:
        # Add ground truth
        ax.axhline(y=prop_Ref1, xmin=0, xmax=1, ls='--',
                   label="Ground Truth: {:3.2}".format(prop_Ref1))

    # Add confidence intervals
    x = ax.get_xticks()
    y = df_bs.mean()
    errors = np.zeros(shape=(2, len(methods)))
    for midx, method in enumerate(methods):
        nobs = len(df_bs[method])
        count = int(np.mean(df_bs[method])*nobs)
        ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method='normal')
        # ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')# ax.plot(x, y, marker='o', ls='', markersize=20)
        errors[0, midx] = y[midx] - ci_low1
        errors[1, midx] = ci_upp1 - y[midx]

    # Add white border around error bars
    # ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c='w', lw=8, capsize=12, capthick=8)
    ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c=c, lw=4, capsize=10, capthick=4, label="Confidence Intervals ({:3.1%})".format(1-alpha))
    #f, ax = plt.subplots()
    #ax = sns.pointplot(data=df_bs, join=False, ci=100*(1-alpha), capsize=.2)
    #sns.despine(top=True, bottom=True, trim=True)
    #plt.savefig('figs/conf_int_{}.png'.format(data_label))

    #yticks = ax.get_yticks()
    #ax.set_yticks(yticks)

    ax.yaxis.tick_right()
    #plt.setp(ax, yticks=yticks)
    #ax.yaxis.set_ticks_position('right')

    #ax.set_xticks([])
    #ax.set_xticklabels(list(methods))
    ax.legend()
    # plt.savefig('figs/violins_{}.png'.format(data_label))

    if False:
        # Plot swarm box
        f, ax = plt.subplots()
        ax = sns.boxplot(data=df_bs)
        ax = sns.swarmplot(data=df_bs, color=".25")
        sns.despine(top=True, bottom=True, trim=True)
        plt.savefig('figs/boxes_{}.png'.format(data_label))







# NOTE: KDEs are very expensive when large arrays are passed to score_samples
# Increasing the tolerance: atol and rtol speeds the process up significantly

#if __name__ == "__main__":

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message="The 'normed' kwarg is deprecated")

#    mpl.style.use('seaborn')
# plt.style.use('seaborn-white')
mpl.rc('figure', figsize=(10, 8))
mpl.rc('font', size=14)
mpl.rc('axes', titlesize=14)     # fontsize of the axes title
mpl.rc('axes', labelsize=14)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=12)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=12)    # fontsize of the tick labels
mpl.rc('legend', fontsize=11)    # legend fontsize
mpl.rc('figure', titlesize=14)  # fontsize of the figure title


# ---------------------------- Define constants ------------------------------

verbose = False
plot_results = False
out_dir = "sweep_results"

# TODO: Reimplement this
adjust_excess = True

seed = 42
bootstraps = 10
sample_size = -1  # 1000
alpha = 0.05

KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# ----------------------------------------------------------------------------

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Set random seed
np.random.seed(seed)
# rng = np.random.RandomState(42)

np.seterr(divide='ignore', invalid='ignore')

if False:
    #metric = 'T1GRS'
    data_label ='Diabetes'
    (scores, bins, means, medians, prop_Ref1) = load_diabetes_data('T1GRS')
else:
    data_label ='Renal'
    (scores, bins, means, medians, prop_Ref1) = load_renal_data()

# plot_distributions(scores, bins, data_label)  #KDE_kernel, bins['width'])

#for (scores, bins, means, median) in ...
# pe.plot_kernels(scores, bins)

if adjust_excess:
    adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
else:
    adjustment_factor = 1.0

methods = OrderedDict([("Means", {'Ref1': means['Ref1'],
                                  'Ref2': means['Ref2']}),
                       ("Excess", {"Median_Ref1": medians["Ref1"],
                                   "Median_Ref2": medians["Ref2"],
                                   "adjustment_factor": adjustment_factor}),
                       ("EMD", True),
                       ("KDE", {'kernel': KDE_kernel,
                                'bandwidth': bins['width']})])

FRESH_DATA = True

res_file = '{}/pe_results_{}'.format(out_dir, data_label)

if FRESH_DATA:
    print("Running mixture analysis with {} scores...".format(data_label))
    t = time.time()  # Start timer

    df_pe = pe.analyse_mixture(scores, bins, methods,
                               bootstraps=bootstraps, sample_size=sample_size,
                               alpha=alpha, true_prop_Ref1=prop_Ref1)

    elapsed = time.time() - t
    print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

    # Save results
    df_pe.to_pickle(res_file)
else:
    if os.path.isfile(res_file):
        df_pe = pd.read_pickle(res_file)

if FRESH_DATA:
    exec(open("./bootstrap.py").read())

sns.set_style("ticks")

fig = plt.figure(figsize=(12,8))
gs = plt.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0.2)

#sns.set_style("ticks")
ax_dists = fig.add_subplot(gs[-1,-1])
plot_distributions(scores, bins, data_label, ax=ax_dists)

#sns.set_style("whitegrid")
ax_ci = fig.add_subplot(gs[0,-1])
with sns.axes_style("whitegrid"):
    #plot_bootstraps(df_pe, prop_Ref1, ax_ci, ylims=(0, 0.12))
    plot_bootstraps(df_pe, prop_Ref1, ax_ci, ylims=None)

#fig = plt.figure(figsize=(12,8))
#plot_bootstraps(df_pe, prop_Ref1) #, None, ylims=(0, 0.12))
#plt.savefig('figs/violins_{}.png'.format(data_label))


# Grid Search Plots

# Configure plots
PLOT_RELATIVE_ERROR = False
PLOT_ABSOLUTE_ERROR = True
PLOT_STANDARD_DEVIATION = True
ADJUST_EXCESS = True

LINEAR_COLOURBAR = True
ABSOLUTE_ERROR = True
LABELS = ['Means', 'Excess', 'EMD', 'KDE']
COLOURS = ['r', 'g', 'b', 'k']
average = np.mean
deviation = np.std

METRICS = ['T1GRS', 'T2GRS']

#for metric in METRICS:
#    plot_bootstrap_results(metric)

ax_grid_Means = fig.add_subplot(gs[0,0], xticklabels=[])
ax_grid_Excess = fig.add_subplot(gs[0,1], xticklabels=[], yticklabels=[])
ax_grid_EMD = fig.add_subplot(gs[1,0])
ax_grid_KDE = fig.add_subplot(gs[1,1], yticklabels=[])

axes = np.array([[ax_grid_Means, ax_grid_Excess], [ax_grid_EMD, ax_grid_KDE]])
plot_bootstrap_results(out_dir, data_label, fig, axes)  # "T1GRS"



plt.tight_layout()
plt.savefig('figs/compund_{}.png'.format(data_label))




if False and plot_results:
    # TODO: Make into a function
    plt.figure()
    fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False)

    axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=labels[:-1])
    legend = axP.legend(facecolor='grey')
    #legend.get_frame().set_facecolor('grey')
    axP.set_title('Proportions of Type 1 and Type 2 vs {}'.format(tag))

    plt.sca(axM)
    res_mix.plot_fit()

    axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")

    plt.sca(axR)
    res_mix.plot_residuals()

    #plt.sca(axI)
    axI.plot(x, kde1, label='Type 1')
    axI.plot(x, kde2, label='Type 2')
