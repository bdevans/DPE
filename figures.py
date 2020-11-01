#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:04:57 2018

Generate manuscript figures.

@author: ben
"""

import os
import time
# from collections import OrderedDict
import warnings

import pandas as pd
import numpy as np
# import scipy as sp
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from statsmodels.stats.proportion import proportion_confint
import tqdm

import proportion_estimation as pe
from datasets import (load_diabetes_data, load_renal_data, load_coeliac_data)


# ---------------------------- Define constants ------------------------------

# METHODS_ORDER = ["Excess", "Means", "EMD", "KDE"]

# out_dir = "results_test"
# out_dir = "results_100_100"
# out_dir = "results/m1000_b0"
# out_dir = "results/m1000_b10"
# out_dir = "results/m100_b1000"
# out_dir = "results/m10_b100"
# out_dir = "results/m100_b100"
# out_dir = "results/m1000_b100"
out_dir = "results/s1000_m0_b0"
fig_dir = os.path.join(out_dir, "figs")

FRESH_DATA = True  # False  # CAUTION!
seed = 42
n_boot = 100
n_mix = 100
correction = False  # True  # Flag to use bias correction: corrected = 2 * pe_point - mean(pe_boot)
sample_size = 1000  # -1
n_seeds = 1
selected_mix = 0
alpha = 0.05
CI_METHOD = "bca"  # "experimental"  # "stderr" # "centile" "jeffreys"
# normal : asymptotic normal approximation
# agresti_coull : Agresti-Coull interval
# beta : Clopper-Pearson interval based on Beta distribution
# wilson : Wilson Score interval
# jeffreys : Jeffreys Bayesian Interval
# binom_test : experimental, inversion of binom_test
# http://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html

application_xlims = {'Diabetes': None, 'Coeliac': (0, 0.3)}
verbose = False

# TODO: Reimplement this
adjust_excess = False
KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# Grid Search Plots
# Configure plots

LINEAR_COLOURBAR = True
ABSOLUTE_ERROR = False
# LABELS = ['Means', 'Excess', 'EMD', 'KDE']
# COLOURS = ['r', 'g', 'b', 'k']
average = np.mean
deviation = np.std

# METRICS = ['T1GRS', 'T2GRS']


# ----------------------------------------------------------------------------

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message="The 'normed' kwarg is deprecated")

# TODO: Move to __main__
#    mpl.style.use('seaborn')
# plt.style.use('seaborn-white')
mpl.rc('figure', figsize=(10, 8))
mpl.rc('font', size=14)
mpl.rc('axes', titlesize=14)    # fontsize of the axes title
mpl.rc('axes', labelsize=14)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=12)   # fontsize of the tick labels
mpl.rc('ytick', labelsize=12)   # fontsize of the tick labels
mpl.rc('legend', fontsize=11)   # legend fontsize
mpl.rc('figure', titlesize=14)  # fontsize of the figure title
mpl.rc('lines', linewidth=2)
mpl.rc('figure', dpi=100)
mpl.rc('savefig', dpi=600)
mpl.rc('mpl_toolkits', legacy_colorbar=False)  # Supress MatplotlibDeprecationWarning


def SecToStr(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u'%d:%02d:%02d' % (h, m, s)


def plot_kernels(scores, bins):

    fig, axes = plt.subplots(len(scores), 1, sharex=True)
    X_plot = bins['centers'][:, np.newaxis]
    for (label, data), ax in zip(scores.items(), axes):
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            kde = pe.fit_kernel(data, bins['width'], kernel)
            ax.plot(X_plot[:, 0], np.exp(kde.score_samples(X_plot)), '-',
                    label="kernel = '{0}'; bandwidth = {1}".format(kernel, bins['width']))
        ax.legend(loc='upper left')
        ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
        ax.set_ylabel(label)
    return (fig, axes)


def load_accuracy(out_dir, label):

    PROPORTIONS = np.load('{}/proportions_{}.npy'.format(out_dir, label))
    SAMPLE_SIZES = np.load('{}/sample_sizes_{}.npy'.format(out_dir, label))
    # PROPORTIONS_Ref2 = PROPORTIONS_Ref1[::-1]

    # Dictionary of p1 errors
    point_estimates = {}
    boots_estimates = {}

    for method in pe._ALL_METHODS_:
        point_file = '{}/point_{}_{}.npy'.format(out_dir, method, label)
        boots_file = '{}/boots_{}_{}.npy'.format(out_dir, method, label)
        if os.path.isfile(point_file):
            point_estimates[method] = np.load(point_file)
        if os.path.isfile(boots_file):
            boots_estimates[method] = np.load(boots_file)

    return point_estimates, boots_estimates, PROPORTIONS, SAMPLE_SIZES


def get_error_bars(df_pe, correction=False, average=np.mean, alpha=0.05, ci_method="bca"):
    """df: columns are method names"""

    #methods = list(df.columns)
    #n_est, n_methods = df.shape
    n_methods = len(df_pe.columns)
    errors = np.zeros(shape=(2, n_methods))
    centres = np.zeros(n_methods)

    for m, method in enumerate(df_pe):  # enumerate(methods):

        boot_values = df_pe.iloc[1:, m]

        if correction:
            centres[m] = 2 * df_pe.iloc[0, m] - np.mean(boot_values)
            boot_values = 2 * df_pe.iloc[0, m] - boot_values
        else:
            centres[m] = df_pe.iloc[0, m]
            # boot_values = df_pe.iloc[1:, m]

        ci_low, ci_upp = pe.calc_conf_intervals(boot_values, correction=False,  #initial=centre,
                                                average=average, alpha=alpha,
                                                ci_method=ci_method)
        errors[0, m] = centres[m] - ci_low
        errors[1, m] = ci_upp - centres[m]

    return (errors, centres)


# TODO: Plot only one colourbar per row: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html
def plot_accuracy(estimates, proportions, sample_sizes, label, fig, ax,
                  shading_levels=None, contour_levels=[0.02],
                  title=True, cbar=True):

    if not ax:
        fig, ax = plt.subplots()

    if estimates[label].ndim > 3:  # Take mean over bootstraps first
        average_error = average(np.mean(estimates[label], axis=3), axis=2) - proportions
    else:
        average_error = average(estimates[label], axis=2) - proportions

    if ABSOLUTE_ERROR:
        average_error = np.abs(average_error)

    # Plot shaded regions for each method on individual subplots
    if not shading_levels:
        if LINEAR_COLOURBAR:
            if ABSOLUTE_ERROR:
                SHADING_LEVELS = np.arange(0, 0.051, 0.005)
                SHADING_TICKS = np.linspace(0, 0.05, 6)
            else:
                SHADING_LEVELS = np.arange(-0.05, 0.051, 0.005)
                # SHADING_LABELS = ["{:.2f}".format(tick) for tick in SHADING_LEVELS[::2]]  # ['< -1', '0', '> 1']
                # ['-0.05', '-0.04', '-0.03', '-0.02', '-0.01', '0.00', '0.01', '0.02', '0.03', '0.04', '0.05']
                SHADING_TICKS = np.linspace(-0.05, 0.05, 11)

        else:
            SHADING_LEVELS = np.logspace(-4, -1, num=19)
            np.logspace(-4, -1, num=4)
    else:
        SHADING_LEVELS = shading_levels
        SHADING_TICKS = ["{:.2f}".format(tick) for tick in SHADING_LEVELS[::2]]
    # SHADING_LABELS = ["{:.2f}".format(tick) for tick in SHADING_TICKS]

    if ABSOLUTE_ERROR:
        cmap = 'viridis_r'
        colour = [sns.color_palette()[6]]
    else:
        #cmap = "RdBu_r" # mpl.colors.ListedColormap(sns.color_palette("RdBu", n_colors=len(SHADING_LEVELS)))
        #cmap = "bwr"
        cmap = "seismic"
        #cmap = "PuOr"
        colour = [sns.color_palette()[2]]

    if LINEAR_COLOURBAR:
        if ABSOLUTE_ERROR:
            CS = ax.contourf(proportions, sample_sizes, average_error,
                             SHADING_LEVELS, cmap=cmap, extend='max')
        else:
            # CS = ax.contourf(proportions, sample_sizes, average_error,
            #                  SHADING_LEVELS, cmap=cmap, extend='both')  # , vmin=-.05, vmax=.05)
            CS = ax.imshow(average_error, SHADING_LEVELS, cmap=cmap, vmin=-.05, vmax=.05)
    else:
        CS = ax.contourf(proportions, sample_sizes, average_error,
                         SHADING_LEVELS, cmap=cmap,
                         locator=mpl.ticker.LogLocator())  # , extend='max'

    # Plot contours from absolute deviation (even if heatmap is +/-)
    ax.contour(proportions, sample_sizes, np.abs(average_error),
               contour_levels, colors=colour, linewidths=mpl.rcParams['lines.linewidth']+1)

    if title:
        ax.set_title(label)

    if cbar:
        cb = fig.colorbar(CS, ax=ax, ticks=SHADING_TICKS)  # , norm=mpl.colors.LogNorm())
        # cb.ax.set_yticklabels(SHADING_LABELS)

    return CS


def plot_deviation(estimates, proportions, sample_sizes, label, fig, ax,
                   shading_levels=np.arange(0.01, 0.1001, 0.005),
                   contour_levels=[0.05], title=True, cbar=True):

    # np.arange(0.005, 0.1001, 0.005)
    if not ax:
        fig, ax = plt.subplots()

    colour = [sns.color_palette()[-3]]

    if estimates[label].ndim > 3:  # Take mean over bootstraps first
        bs_dev = deviation(np.mean(estimates[label], axis=3), axis=2)
    else:
        bs_dev = deviation(estimates[label], axis=2)
    if title:
        ax.set_title(label)
    CS = ax.contourf(proportions, sample_sizes, bs_dev,
                     shading_levels, cmap='viridis_r', extend='both')

    ax.contour(proportions, sample_sizes, bs_dev, contour_levels,
               colors=colour, linewidths=mpl.rcParams['lines.linewidth']+1)
    if cbar:
        fig.colorbar(CS, ax=ax)

    return CS


def plot_characterisation(estimates, proportions, sample_sizes,
                          figsize=(16, 8), cl=None):
    if cl is None:
        cl = [0.02]

    fig = plt.figure(figsize=figsize)
    # gs = plt.GridSpec(nrows=2, ncols=4, hspace=0.15, wspace=0.15)

    # rect=[left, bottom, width, height] or 111
    left = 0.06
    bottom = 0.10
    width = 0.88
    height = 0.84
    grid = AxesGrid(fig, rect=[left, bottom, width, height], aspect=False,  # similar to subplot(122)
                    nrows_ncols=(2, len(pe._ALL_METHODS_)),
                    axes_pad=0.15,
                    label_mode="L",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad="8%",
                    )

    x_half_width = (proportions[1] - proportions[0]) / 2
    y_half_width = (sample_sizes[1] - sample_sizes[0]) / 2
    extent = (proportions[0]-x_half_width, proportions[-1]+x_half_width,
              sample_sizes[0]-y_half_width, sample_sizes[-1]+y_half_width)

    for m, method in enumerate(pe._ALL_METHODS_):  # enumerate(methods):

        # Plot average accuracy across mixtures
        # ax_acc = fig.add_subplot(gs[0, m], xticklabels=[])
        ax_acc = grid[m]
        if m == 0:
            ax_acc.set_ylabel('Sample size ($n$)')
        # else:
        #     ax_acc.set_yticklabels([])

        # hm = plot_accuracy(estimates, proportions, sample_sizes, method, fig, ax_acc, contour_levels=cl, cbar=False)

        label = method
        if estimates[label].ndim > 3:  # Take mean over bootstraps first
            average_error = average(np.mean(estimates[label], axis=3), axis=2) - proportions
        else:
            average_error = average(estimates[label], axis=2) - proportions

#        shading_levels = np.arange(-0.05, 0.051, 0.005)
        shading_levels = np.linspace(-0.05, 0.05, num=21, endpoint=True)
#        shading_levels = np.arange(-0.05, 0.051, 0.01)
        shading_ticks = np.linspace(-0.05, 0.05, 11, endpoint=True)
        cmap = "seismic"
        cmap = plt.cm.get_cmap("seismic", len(shading_levels)-1)  # discrete colours

        hm = ax_acc.imshow(average_error, cmap=cmap, vmin=min(shading_levels), vmax=max(shading_levels), origin='lower', extent=extent, aspect=0.0004)  #shading_levels,
        # hm = ax_acc.contourf(proportions, sample_sizes, np.random.randn(len(sample_sizes), len(proportions)))
        # hm = ax_acc.imshow(np.random.randn(len(sample_sizes), len(proportions)), origin='lower', extent=extent, aspect=0.001)
        ax_acc.set_xlim(extent[:2])
        ax_acc.set_ylim(extent[2:])
        ax_acc.set_title(method)
        if m % len(pe._ALL_METHODS_) == 3:
            cax = grid.cbar_axes[0]
            cax.colorbar(hm, ticks=shading_ticks)  # , extend='both')  # TODO: Fix!
            cax.toggle_label(True)
            cax.axis[cax.orientation].set_label("Average Deviation from $p_C$")

        # Plot deviation across mixtures
        # ax_dev = fig.add_subplot(gs[1, m])
        ax_dev = grid[m+len(pe._ALL_METHODS_)]
        if m == 0:
            ax_dev.set_ylabel('Sample size ($n$)')
        # else:
        #     ax_dev.set_yticklabels([])
        ax_dev.set_xlabel('$p_C^*$')  # '$p_1^*$'
        # hm = plot_deviation(estimates, proportions, sample_sizes, method, fig, ax_dev, title=False, cbar=False)

        # SD values
#        vmin = 0.01
#        vmax = 0.1

        # Model variability
        def deviation(estimates, axis=None, alpha=0.05):
            ci_low, ci_upp = np.percentile(estimates, [100*alpha/2, 100-(100*alpha/2)], axis=axis)
            return ci_upp - ci_low

#        vmin = 0
#        vmax = 0.25


        if estimates[label].ndim > 3:  # Take mean over bootstraps first
            bs_dev = deviation(np.mean(estimates[label], axis=3), axis=2)
        else:
            bs_dev = deviation(estimates[label], axis=2)
        cmap = "viridis_r"

#        shading_levels = np.arange(0.0, 0.4001, 0.05)
        shading_levels = np.linspace(0.0, 1.0, num=21, endpoint=True)
#        shading_levels = np.arange(0.0, 0.4001, 0.05)
        cmap = plt.cm.get_cmap(cmap, len(shading_levels)-1)  # discrete colours

        hm = ax_dev.imshow(bs_dev, cmap=cmap, vmin=min(shading_levels), vmax=max(shading_levels), origin='lower', extent=extent, aspect=0.0004)
        # hm = ax_dev.contourf(proportions*1000, sample_sizes, bs_dev, shading_levels, vmin=.01, vmax=.10, cmap=cmap, extend='both')

        # hm = ax_dev.contourf(proportions, sample_sizes, np.random.randn(len(sample_sizes), len(proportions)))
        # hm = ax_dev.imshow(np.random.randn(len(sample_sizes), len(proportions)), origin='lower', extent=extent, aspect=0.001)


        ax_dev.set_xlim(extent[:2])
        ax_dev.set_ylim(extent[2:])

        # Show only 1 sf for 0.00 and 1.00 to avoid crowding
        xticks = [0, 0.25, 0.5, 0.75, 1]
        ax_dev.set_xticks(xticks)
#        xticklabels = ax_dev.get_xticks().tolist()  # [label.get_text() for label in ax_dev.get_xticklabels()]
#        xticklabels[0] = "0"
#        xticklabels[-1] = "1"
        ax_dev.set_xticklabels([str(tick) for tick in xticks])
        if m % len(pe._ALL_METHODS_) == 3:
            # print(method)
            # cax = mpl.colorbar.Colorbar(ax, hm)
            cax = grid.cbar_axes[1]
            cax.colorbar(hm)  # , extend='max')  # TODO: Fix!
            cax.toggle_label(True)
            cax.axis[cax.orientation].set_label("Model variability")


    # grid.axes_llc.set_xlim(extent[:2])
    # grid.axes_llc.set_ylim(extent[2:])

    return fig


def plot_distributions(scores, bins, data_label, norm=False, despine=True, ax=None):

    if not ax:
        f, ax = plt.subplots()

    palette = ['#6f92f3', '#aac7fd', '#bbbbbb', '#f7b89c', '#e7745b']
    # TODO: Melt data together so the y-axis is normalised properly?

    with sns.axes_style("ticks") and warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sns.distplot(scores['Ref1'], bins=bins['edges'], norm_hist=norm,
                     label="$R_C: n={:,}$".format(len(scores['Ref1'])),
                     # label="$R_1: n={:,}$".format(len(scores['Ref1'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[-1])
        sns.distplot(scores['Ref2'], bins=bins['edges'], norm_hist=norm,
                     label="$R_N: n={:,}$".format(len(scores['Ref2'])),
                     # label="$R_2: n={:,}$".format(len(scores['Ref2'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[0])
        sns.distplot(scores['Mix'], bins=bins['edges'], norm_hist=norm,
                     label=r"$\tilde{{M}}: n={:,}$".format(len(scores['Mix'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[2])

        if despine:
            sns.despine(top=True, bottom=False, left=False, right=True, trim=True)

    #ax.yaxis.tick_left()
    #ax.yaxis.set_ticks_position('left')

    # if len(indep_vars) > 1:
    #     Use jointplot
    # ax.set_xlabel("GRS")
    ax.set_xlabel("Genetic Risk Score")
    ax.legend()
    # plt.savefig('figs/distributions_{}.png'.format(data_label))


def plot_bootstraps(df_pe, correction=None, initial=True, prop_Ref1=None,
                    ax=None, limits=None, alpha=0.05, ci_method='bca',
                    violins=True, legend=True, orient='v'):

    # c = sns.color_palette()[-3]  # 'gray'
    c = '#999999'
    c_edge = '#777777'

    # df_bs = df_bs[pe._ALL_METHODS_]

    if not ax:
        f, ax = plt.subplots()
    if limits:
        if orient == 'v':
            ax.set_ylim(limits)
        if orient == 'h':
            ax.set_xlim(limits)

    df_point = df_pe.iloc[0, :]
    df_bs = df_pe.iloc[1:, :]
    # TODO: Think...
#    if correction:
#        df_correct = pe.correct_estimate(df_pe)


    # Draw violin plots of bootstraps
    if violins:
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=FutureWarning)
        sns.violinplot(data=df_bs, orient=orient, ax=ax, cut=0, inner=None,
                       palette=sns.color_palette("muted"), saturation=1.0)

    # TODO: This could all be passed in a dict when setting the axis style context
    if orient == 'v':
        sns.despine(ax=ax, top=True, bottom=True, left=False, right=True)#, trim=True)
    elif orient == 'h':
        if True:
            sns.despine(ax=ax, top=True, bottom=False, left=True, right=True)#, trim=True)
        else:
            sns.despine(ax=ax, top=True, bottom=False, left=True, right=False)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    if prop_Ref1:  # Add ground truth
        truth_label = r"$p_C = {:4.3}$".format(prop_Ref1)
        if orient == 'v':
            ax.axhline(y=prop_Ref1, xmin=0, xmax=1, ls='--', c='#aaaaaa',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(prop_Ref1))
        elif orient == 'h':
            ax.axvline(x=prop_Ref1, ymin=0, ymax=1, ls='--', c='#aaaaaa',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(prop_Ref1))

    # Add confidence intervals # TODO: Refactor
#    if orient == 'v':
#        x, y = ax.get_xticks(), df_bs.mean().values
#        means = y
#    elif orient =='h':
#        x, y = df_bs.mean().values, ax.get_yticks()
#        means = x

#    if orient == 'v':
#        x, y = ax.get_xticks(), df_point.iloc[0].values
#        means = y
#    elif orient == 'h':
#        x, y = df_point.iloc[0].values, ax.get_yticks()
#        means = x

    errors, centres = get_error_bars(df_pe, correction=correction)

    if correction:
        if orient == 'v':
            x, y = ax.get_xticks(), centres
            x_init, y_init = x, df_point
            if not violins:
                x = x_init = list(range(len(df_point)))
                ax.set_xticklabels(list(methods))
        elif orient == 'h':
            x, y = centres, ax.get_yticks()
            print(x, y)
            print(type(y))
            x_init, y_init = df_point, y
            if not violins:
                y = y_init = list(range(len(df_point)))
                ax.set_yticklabels(list(methods))
        print(x, y)

        # Plot initial estimate
        if initial:
            # ax.plot(x_init, y_init, 'o', markersize=10, c='#737373')  #(0.25, 0.25, 0.25))
            ax.plot(x_init, y_init, 's', markersize=7, c=c, markeredgecolor=c_edge, label="Initial", zorder=10)

    else:
        if orient == 'v':
            x, y = ax.get_xticks(), df_pe.iloc[0].values
        elif orient == 'h':
            x, y = df_pe.iloc[0].values, ax.get_yticks()

#    if correction:
#        # Plot initial estimate
#        ax.plot(x, y, 'o', markersize=12, c=(0.25, 0.25, 0.25))
#        ax.plot(x, y, 'o', markersize=8, c=c, label="Initial")

    # errors = np.zeros(shape=(2, len(methods)))

#    for midx, method in enumerate(df_bs):  # enumerate(methods):
#
#        # initial = df_point.iloc[0][method]  # Avoid chained indexing
#        # initial = df_point.iloc[0, midx]
#
#        if correction:
#            # Plot initial estimate
#            ax.plot(x, y, fmt='x', markersize=12, c=(0.25, 0.25, 0.25))
#            ax.plot(x, y, fmt='x', markersize=8, c=c, label="Initial")
#
##            centre = df_correct.iloc[0, midx]
##
##        else:
##            centre = df_point.iloc[0, midx]
##
##        ci_low, ci_upp = pe.calc_conf_intervals(df_bs[method], initial=centre, average=np.mean, alpha=alpha, ci_method=ci_method)
##        errors[0, midx] = means[midx] - ci_low
##        errors[1, midx] = ci_upp - means[midx]



    # Add white border around error bars
    # ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c='w', lw=8, capsize=12, capthick=8)

#    error_label = "Confidence Intervals ({:3.1%})".format(1-alpha)
    error_label = f"{1-alpha:3.1%} CI"
#    if correction:
#        error_label += " (Corrected)"

    if orient == 'v':
        ax.errorbar(x=x, y=y, yerr=errors, fmt='none', markersize=14, c=c, lw=2, markeredgecolor=c_edge,
                    capsize=12, capthick=2, label=error_label, zorder=10)

        # Add grey border around error bars
        ax.errorbar(x=x, y=y, yerr=errors, fmt='none', c=c_edge, lw=3, capsize=14, capthick=6)
    elif orient == 'h':
        ax.errorbar(x=x, y=y, xerr=errors, fmt='none', markersize=14, c=c, lw=1.25, markeredgecolor=c_edge,
                    capsize=12, capthick=1, label=error_label, zorder=10)
        
        # Add grey border around error bars
        ax.errorbar(x=x, y=y, xerr=errors, fmt='none', c=c_edge, lw=2, capsize=14, capthick=4)


    if correction:
        # ax.plot(x, y, '*', markersize=14, c=c, markeredgecolor=c_edge, label="Corrected", zorder=20)
        ax.plot(x, y, 'o', markersize=10, c=c, markeredgecolor=c_edge, label="Corrected", zorder=20)
    else:
        ax.plot(x, y, 'o', markersize=6, c=c, markeredgecolor=c_edge, zorder=20)
        ax.plot(x, y, '.', markersize=2, c=c_edge, markeredgecolor=c_edge, label=r"$\hat{p}_C$", zorder=30)

    if orient == 'v':
        ax.yaxis.tick_left()
        # ax.set_ylabel("$p_C$", {"rotation": "horizontal"})  # "$p_1$"
        ax.set_ylabel("Mixture prevalence", {"rotation": "horizontal"})
        ax.set_ylim(0, 1)
        # ax.set_xticks([])  # Remove ticks for method labels
    elif orient == 'h':
        ax.xaxis.tick_bottom()
        # ax.set_xlabel("$p_C$")  # "$p_1$"
        ax.set_xlabel("Mixture prevalence")
        ax.set_xlim(0, 1)
        # ax.set_yticks([])  # Remove ticks for method labels
    #plt.setp(ax, yticks=yticks)
    #ax.yaxis.set_ticks_position('right')

    #ax.set_xticks([])
    #ax.set_xticklabels(list(methods))
    if legend:
        ax.legend()
    # plt.savefig('figs/violins_{}.png'.format(data_label))

    if False:
        # Plot swarm box
        f, ax = plt.subplots()
        ax = sns.boxplot(data=df_bs)
        ax = sns.swarmplot(data=df_bs, color=".25")
        sns.despine(top=True, bottom=True, trim=True)
        plt.savefig(os.path.join(fig_dir, 'boxes_{}.png'.format(data_label)))


def construct_mixture(Ref1, Ref2, p1, size):
    assert(0.0 <= p1 <= 1.0)
    n_Ref1 = int(round(size * p1))
    n_Ref2 = size - n_Ref1

    mix = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                          np.random.choice(Ref2, n_Ref2, replace=True)))
    return mix


def plot_selected_violins(scores, bins, df_point, df_boots, methods, 
                          p_stars, sizes, out_dir, data_label, selected_mix=0,
                          add_ci=True, alpha=0.05, ci_method="bca",
                          correction=False):

    c = sns.color_palette()[-3]  # 'gray'
#    palette=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
#             "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]  # bright
#    palette = sns.color_palette().as_hex()
    palette = sns.color_palette("coolwarm", len(p_stars)+2).as_hex()  # "hls"
    # sns.palplot(sns.diverging_palette(255, 40, s=50, l=70, n=5, center="dark"))
    # palette = sns.diverging_palette(255, 40, s=50, l=70, n=5, center="dark").as_hex()
    if len(palette) % 2 == 1:
        palette[len(palette)//2] = '#bbbbbb'
    print(palette)

    fig_select = plt.figure(figsize=(12, 3*len(sizes)))
    gs = plt.GridSpec(nrows=len(sizes), ncols=2, width_ratios=[2, 3],
                      hspace=0.15, wspace=0.15,
                      left=0.05, right=0.96, bottom=0.08, top=0.97)

    for si, size in enumerate(sizes):
#        ax_vio = fig_select.add_subplot(gs[-(si+1), :-1])
#        ax_mix = fig_select.add_subplot(gs[-(si+1), -1])
        mix_dist_file = f'{out_dir}/mix{selected_mix}_size{size}_{data_label}.pkl'
        if not os.path.isfile(mix_dist_file):
            warnings.warn(f"File not found: {mix_dist_file}")
            return
        df_mixes = pd.read_pickle(mix_dist_file)

        if si == 0:
            # Save base axes
            ax_vio = fig_select.add_subplot(gs[-(si+1), 1:])
            ax_vio_base = ax_vio
#            sns.plt.xlim(0, 1)
#            vio_xlabs = ax_vio_base.get_xticklabels()
            ax_mix = fig_select.add_subplot(gs[-(si+1), 0])
            ax_mix_base = ax_mix
        else:
            ax_vio = fig_select.add_subplot(gs[-(si+1), 1:], sharex=ax_vio_base)
            plt.setp(ax_vio.get_xticklabels(), visible=False)
            ax_mix = fig_select.add_subplot(gs[-(si+1), 0], sharex=ax_mix_base)
            plt.setp(ax_mix.get_xticklabels(), visible=False)

        # Plot constructed mixture distributions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.distplot(scores["Ref2"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True}, # hist=True, norm_hist=True, kde=False)#,
                         label=r"$p_C=0.0\ (R_N)$", ax=ax_mix, color=palette[0])
            for p, p_star in enumerate(p_stars):
                sns.distplot(df_mixes[p_star], bins=bins['edges'], hist=False,
                             label=r"$p_C={}$".format(p_star), ax=ax_mix, color=palette[p+1])  # \tilde{{M}}: n={},
            sns.distplot(scores["Ref1"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True},
                         label=r"$p_C=1.0\ (R_C)$", ax=ax_mix, color=palette[len(p_stars)+1])

        # Plot violins of bootstrapped estimates
        for p, p_star in enumerate(p_stars):

            # Add annotations for p1*
            ax_vio.axvline(x=p_star, ymin=0, ymax=1, ls='--', lw=3, zorder=0, color=palette[p+1])
                       #label="Ground Truth: {:3.2}".format(p_star))

            # Add shading around the true values
            if False:
                shade_span = 0.02
                ax_vio.axvspan(p_star-shade_span, p_star+shade_span, alpha=0.5, zorder=0, color=palette[p+1])


            # Select estimates at p_star and size for all methods
            df_b = df_boots[np.isclose(p_star, df_est['p1*']) &
                            (df_est['Size'] == size) &
                            (df_est["Mix"] == selected_mix)]


            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
    #            g = sns.violinplot(x='Estimate', y='Size', hue='Method', data=df, ax=ax_vio, orient='h', cut=0, linewidth=2)
                sns.violinplot(x='Estimate', y='Method', data=df_b, ax=ax_vio,
                               orient='h', cut=0, linewidth=2, inner=None,
                               color=palette[p+1], saturation=0.7)

#            handles, labels = g.get_legend_handles_labels()
#            g.legend(handles, labels[:len(methods)], title="Method")
            # ax_vio.set_ylabel(r"$n={:,}$".format(size))  # , rotation='horizontal')
            ax_vio.yaxis.set_label_position("right")
            # ax_vio.yaxis.tick_right()
            ax_vio.set_ylabel("")
            ax_vio.set_xlabel("")
#            ax_vio.set_xticklabels([])
            # Remove y axis
#            sns.despine(ax=ax_vio, top=True, bottom=False, left=True, right=True, trim=True)
#            ax_vio.set_yticks([])
#            ax_vio.set_yticklabels([])
#            ax_vio.get_yticklabels().set_visible(False)
            ax_vio.set(xlim=(0, 1))

            if add_ci:  # Add confidence intervals
#                # The true value will be within these bars for 95% of samples (not measures)
#                # For alpha = 0.05, the CI bounds are +/- 1.96*SEM
#                df_means = df_b.groupby('Method').mean()
#                errors = np.zeros(shape=(2, len(methods)))
#                means = []
#                initials = []
#                for midx, method in enumerate(pe._ALL_METHODS_):  # enumerate(methods):
#
#                    mean_est = df_means.loc[method, 'Estimate']
#
#                    initial = df_point[np.isclose(p_star, df_point['p1*'])
#                                       & (df_point['Size'] == size)
#                                       & (df_point["Mix"] == selected_mix)][method].values[0]
#
#                    df_piv = df_b.pivot_table(values="Estimate",
#                                            index=["p1*", "Size", "Mix", "Boot"],
#                                            columns="Method")
#
#                    ci_low, ci_upp = pe.calc_conf_intervals(df_piv[method], initial=initial, average=np.mean, alpha=0.05, ci_method=CI_METHOD)
#
#                    means.append(mean_est)
#                    initials.append(initial)
##                    errors[0, midx] = mean_est - ci_low  # y[midx] - ci_low1
##                    errors[1, midx] = ci_upp - mean_est  # ci_upp1 - y[midx]
#                    errors[0, midx] = initial - ci_low  # y[midx] - ci_low1
#                    errors[1, midx] = ci_upp - initial  # ci_upp1 - y[midx]
#
##                x = means
#                x = initials
#                y = ax_vio.get_yticks()
#
#                # Add grey border around error bars
#                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='none', c=(0.45, 0.45, 0.45), lw=5, capsize=14, capthick=5)
#
#                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='*', markersize=18 ,
#                                c=palette[p+1], lw=2, capsize=12, capthick=2,
#                                label="Confidence Intervals ({:3.1%})".format(1-alpha),
#                                markeredgecolor=(0.45, 0.45, 0.45))


                ##### NEW METHOD #####

                # Estract point estimates for the particular hyperparameters
                df_p = df_point[np.isclose(p_star, df_point['p1*'])
                                & (df_point['Size'] == size)
                                & (df_point["Mix"] == selected_mix)]

                df_p_piv = df_p.drop(columns=["p1*", "Size", "Mix"])

                df_b_piv = df_b.pivot_table(values="Estimate",
                                            index=["p1*", "Size", "Mix", "Boot"],
                                            columns="Method")

                df_b_piv = df_b_piv[df_p_piv.columns]  # Manually sort before merge
                df_pe = pd.concat([df_p_piv, df_b_piv], ignore_index=True)
                errors, centres = get_error_bars(df_pe, correction=correction)


                if correction:
                    x, y = centres, ax_vio.get_yticks()
                    x_init, y_init = np.squeeze(df_p_piv), y

                    # Plot initial estimate
                    # ax_vio.plot(x_init, y_init, 'o', markersize=10, c=(0.45, 0.45, 0.45))
                    ax_vio.plot(x_init, y_init, 'o', c=palette[p+1],
                                markersize=9, markeredgecolor=(0.45, 0.45, 0.45), label="Initial", zorder=10)

                else:
                    x, y = df_pe.iloc[0].values, ax_vio.get_yticks()

                # Add grey border around error bars
                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='none',
                                c=(0.45, 0.45, 0.45), lw=5, capsize=14, capthick=5)

                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='none', c=palette[p+1],
                                markersize=12, lw=2, capsize=12, capthick=2,
                                markeredgecolor=(0.45, 0.45, 0.45),
                                label=f"Confidence Intervals ({1-alpha:3.1%})")

                if correction:
                    ax_vio.plot(x, y, '*', c=palette[p+1], markersize=12, markeredgecolor=(0.45, 0.45, 0.45), label="Corrected", zorder=20)
                else:
                    ax_vio.plot(x, y, 'o', c=palette[p+1], markersize=9, markeredgecolor=(0.45, 0.45, 0.45), zorder=20)  # label=r"$\hat{p}_C$", 
                    ax_vio.plot(x, y, '.', c=(0.45, 0.45, 0.45), markersize=2, markeredgecolor=(0.45, 0.45, 0.45), label=r"$\hat{p}_C$", zorder=30)


        if si == len(sizes)-1:  # Top row
            handles, labels = ax_mix.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax_mix.legend(handles, labels)
            # ax_mix.legend()
        else:
            if ax_mix.get_legend():
                # Remove legend and label yaxis instead
                ax_mix.get_legend().set_visible(False)

        # Remove y axis
        # Set ticks at the ground truth values and do not trim
        ax_vio.set_xticks([0, *p_stars, 1])
        # Do not show labels for 0.00 and 1.00 to avoid crowding
#        vio_xlabs = ax_vio_base.get_xticklabels()
        vio_xlabs = ax_vio.get_xticks().tolist()
#        vio_xlabs[0] = ''
#        vio_xlabs[-1] = ''
        ax_vio.set_xticklabels(vio_xlabs)
#        ax_vio_base.set_xticklabels()
        sns.despine(ax=ax_vio, top=True, bottom=False, left=True, right=True, trim=False)
#        ax_vio.set_yticks([])
#        ax_vio.set_yticklabels([])
#        ax_mix.set_ylabel(r"$n={}$".format(size), rotation='horizontal')
        ax_mix.set_xlabel("")
        # Remove y axis
        sns.despine(ax=ax_mix, top=True, bottom=False, left=True, right=True, trim=True)
        ax_mix.set_yticks([])
        ax_mix.set_yticklabels([])
        # ax_mix.set_ylabel("")
        ax_mix.set_ylabel(r"$n={:,}$".format(size), fontweight='heavy', fontsize=16)
#    g.invert_yaxis()
    # ax_vio_base.set_xlabel(r"Estimated prevalence ($\hat{p}_C$)")  # $p_1$
    ax_vio_base.set_xlabel("Mixture prevalence")
    # ax_mix_base.set_xlabel("GRS")
    ax_mix_base.set_xlabel("Genetic Risk Score")
#    ax_vio_base.set_xticks()
#    plt.setp(ax_vio.get_xticklabels(), visible=True)
#    plt.tight_layout()
    fig_select.savefig(os.path.join(fig_dir, f'violin_selection_{selected_mix}_{data_label}.png'))
    fig_select.savefig(os.path.join(fig_dir, f'violin_selection_{selected_mix}_{data_label}.svg'), transparent=True)


def plot_ROC(scores, bins, title=None, ax=None):
    """Plot the Reciever Operator characteristic

    Args:

    """
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    if not ax:
        fig, ax = plt.subplots()
    plt.sca(ax)

    # scores, bins
    # method = 'fd'
    # all_refs = [*scores["Ref1"], *scores["Ref2"]]
    # # probas_, edges_r = np.histogram(all_refs, bins=method, range=(bins["min"], bins["max"]), density=True)
    # probas_, edges_a = np.histogram(all_refs, bins=bins["edges"], density=True)


    # TPR = TP / P
    # FPR = FP / N
    # x=FPR, y=TPR
    density = False
    # p = 1. * np.arange(len(all_refs)) / (len(all_refs) - 1)
    hist_1, _ = np.histogram(scores["Ref1"], bins=bins["edges"], density=density)
    hist_2, _ = np.histogram(scores["Ref2"], bins=bins["edges"], density=density)
    if scores["Ref1"].mean() > scores["Ref2"].mean():
        hist_p = hist_1
        hist_n = hist_2
    else:
        hist_p = hist_2
        hist_n = hist_1

    cum_p = np.cumsum(hist_p)  # Ref1 := cases := positive
    cond_P = hist_p.sum()
    # print(hist_c)
    # print(cum_c)
    tp = cond_P - cum_p  # Must subtract from P since cumsum grows the opposite way
    
    cum_n = np.cumsum(hist_n)  # Ref2 := non-cases := negative
    cond_N = hist_n.sum()
    # print(hist_n)
    # print(cum_n)
    tn = cum_n

    # if scores["Ref1"].mean() > scores["Ref2"].mean():
    #     tp = np.flip(tp)
    #     tn = np.flip(tn)

    # print(bins['centers'])
    # print(tp)
    # print(tn)

    # Assume mean(GRS_c) > mean(GRS_n)
    tpr = tp / cond_P
    fpr = 1 - (tn / cond_N)
    # print(tpr)
    # print(fpr)

    # fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
    # fpr = np.r_[0, fpr]
    # tpr = np.r_[0, tpr]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    if not title:
        plt.title('Receiver operating characteristic')
    else:
        plt.title(title)
    plt.legend(loc="lower right")
    ax.set(aspect="equal")

    return roc_auc

# NOTE: KDEs are very expensive when large arrays are passed to score_samples
# Increasing the tolerance: atol and rtol speeds the process up significantly



if __name__ == "__main__":

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Set random seed
    np.random.seed(seed)
    # rng = np.random.RandomState(42)
    np.seterr(divide='ignore', invalid='ignore')

    # Plot ROC curves
    fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(18, 12))
    # fig, axes = plt.subplots(3,3)
    (scores, bins, means, medians, prop_Ref1) = load_diabetes_data('T1GRS')
    # scores['Ref1'] = np.append(scores['Ref1'], np.random.randn(100000)*0.1+0.8)
    plot_ROC(scores, bins, title='Diabetes: T1GRS', ax=axes[0, 0])
    plot_distributions(scores, bins, 'Diabetes: T1GRS', norm=True, despine=False, ax=axes[1, 0])
    (scores, bins, means, medians, prop_Ref1) = load_diabetes_data('T2GRS')
    # scores['Ref1'] = np.append(scores['Ref1'], np.random.randn(100000)+5)
    plot_ROC(scores, bins, title='Diabetes: T2GRS', ax=axes[0, 1])
    plot_distributions(scores, bins, 'Diabetes: T2GRS', norm=True, despine=False, ax=axes[1, 1])
    (scores, bins, means, medians, prop_Ref1) = load_renal_data()
    # scores['Ref1'] = np.append(scores['Ref1'], np.random.randn(100000)+10)
    plot_ROC(scores, bins, title='Renal', ax=axes[0, 2])
    plot_distributions(scores, bins, 'Renal', norm=True, despine=False, ax=axes[1, 2])
    fig.savefig(os.path.join(fig_dir, 'roc_Diabetes.png'))
    # exit()

    for data_label, data in [("Diabetes", load_diabetes_data('T1GRS')),
                            #  ("Renal", load_renal_data()),
                             ("Coeliac", load_coeliac_data())]:

        (scores, bins, means, medians, prop_Ref1) = data

        if adjust_excess:
            adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
        else:
            adjustment_factor = 1.0

        methods = {method: True for method in pe._ALL_METHODS_}
        res_file = '{}/pe_results_{}.pkl'.format(out_dir, data_label)

        if FRESH_DATA:  # or True:
            print("Running mixture analysis on {} scores...".format(data_label), flush=True)
            t = time.time()  # Start timer

            df_pe = pe.analyse_mixture(scores, bins, methods,
                                       n_boot=n_boot, boot_size=-1, n_mix=n_mix, # boot_size=sample_size,
                                       alpha=alpha, true_p1=prop_Ref1, 
                                       correction=correction, n_jobs=-1,  # Previously correction defaulted to False
                                       logfile=f"{out_dir}/pe_{data_label}.log")

            elapsed = time.time() - t
            print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

            # Save results
            df_pe.to_pickle(res_file)
        else:
            print("Loading {} analysis...".format(data_label), flush=True)
            if os.path.isfile(res_file):
                df_pe = pd.read_pickle(res_file)
            else:
                warnings.warn("Missing data file: {}".format(res_file))
                break

        #if FRESH_DATA:
        #    exec(open("./bootstrap.py").read())

        sns.set_style("ticks")

        # Load bootstraps of accurarcy data
        (point_estimates, boots_estimates, proportions, sample_sizes) = load_accuracy(out_dir, data_label)

        # Plot point estimates of p1
        if bool(point_estimates):
            print("Plotting characterisation of {} scores...".format(data_label), flush=True)
            fig = plot_characterisation(point_estimates, proportions, sample_sizes)
            fig.savefig(os.path.join(fig_dir, 'point_characterise_{}.png'.format(data_label)))
            fig.savefig(os.path.join(fig_dir, 'point_characterise_{}.svg'.format(data_label)), transparent=True)

        # Plot bootstrapped estimates of p1
        if False:  # bool(boots_estimates):
            print("Plotting bootstrapped characterisation of {} scores...".format(data_label), flush=True)
            fig = plot_characterisation(boots_estimates, proportions, sample_sizes)
            fig.savefig(os.path.join(fig_dir, 'boots_characterise_{}.png'.format(data_label)))

        # Plot violins for a set of proportions
        # p_stars = [0.05, 0.25, 0.50, 0.75, 0.95]
        # sizes = [100, 500, 1000, 5000, 10000]

        p_stars = [0.25, 0.50, 0.75]
        sizes = [500, 1000, 5000]
        # n_boot = 5

        # Generate multiple mixes
        point_estimates_res_file = f'{out_dir}/pe_stack_analysis_point_{data_label}.pkl'
        boot_estimates_res_file = f'{out_dir}/pe_stack_analysis_{data_label}.pkl'
        if FRESH_DATA:  # or True:
            print(f"Running mixture analysis with {data_label} scores...", flush=True)
            t = time.time()  # Start timer

            # Split the references distributions to ensure i.i.d. data for 
            # constructing the mixtures and estimating them.
            n_RefC, n_RefN = len(scores['Ref1']), len(scores['Ref2'])
            partition_RefC, partition_RefN = n_RefC//2, n_RefN//2
            inds_RefC = np.random.permutation(n_RefC)
            inds_RefN = np.random.permutation(n_RefN)
            hold_out_scores = {'Ref1': scores['Ref1'][inds_RefC[:partition_RefC]],
                               'Ref2': scores['Ref2'][inds_RefN[:partition_RefN]]}
            violin_scores = {'Ref1': scores['Ref1'][inds_RefC[partition_RefC:]],
                             'Ref2': scores['Ref2'][inds_RefN[partition_RefN:]]}

            dfs_point = []
            dfs_boot = []

            size_bar = tqdm.tqdm(sizes, dynamic_ncols=True)
            for s, size in enumerate(size_bar):
                size_bar.set_description(f"Size = {size:6,}")
                Mixtures = {mix: {} for mix in range(n_seeds)}

                for mix in tqdm.trange(n_seeds, dynamic_ncols=True, desc=" Mix"):
                    mix_dist_file = f'{out_dir}/mix{mix}_size{size}_{data_label}.pkl'

                    prop_bar = tqdm.tqdm(p_stars, dynamic_ncols=True)
                    for p, p_star in enumerate(prop_bar):
                        prop_bar.set_description(f" p1* = {p_star:6.2f}")

                        violin_scores['Mix'] = construct_mixture(hold_out_scores['Ref1'], hold_out_scores['Ref2'], p_star, size)
                        Mixtures[mix][p_star] = violin_scores['Mix']
                        df_cm = pe.analyse_mixture(violin_scores, bins, methods,
                                                   n_boot=n_boot, boot_size=size,
                                                   n_mix=n_mix,
                                                   alpha=alpha, true_p1=p_star,
                                                   correction=correction,  # Previously correction defaulted to False
                                                   n_jobs=-1, verbose=0)
                        df_point = df_cm.iloc[[0]].copy()
                        df_point['Size'] = size
                        df_point['p1*'] = p_star
                        df_point['Mix'] = mix
#                        df_point = df_point.melt(var_name='Method', id_vars=['p1*', 'Size', 'Mix'], value_name='Estimate')
                        dfs_point.append(df_point)

                        df_boots = df_cm.iloc[1:, :].copy()
                        if n_mix > 0:
                            n_bootstraps = n_mix * n_boot
                        else:
                            n_bootstraps = n_boot
                        df_boots['Size'] = size * np.ones(n_bootstraps, dtype=int)
                        df_boots['p1*'] = p_star * np.ones(n_bootstraps)
                        df_boots['Mix'] = mix * np.ones(n_bootstraps, dtype=int)
                        df_boots["Boot"] = list(range(n_bootstraps))
                        df_boots = df_boots.melt(var_name='Method',
                                                 id_vars=['p1*', 'Size', 'Mix', "Boot"],
                                                 value_name='Estimate')
                        dfs_boot.append(df_boots)

                    df_size = pd.DataFrame(Mixtures[mix], columns=p_stars)
                    df_size.to_pickle(mix_dist_file)

            df_point = pd.concat(dfs_point, ignore_index=True)
            df_est = pd.concat(dfs_boot, ignore_index=True)
            elapsed = time.time() - t
            print(f'Elapsed time = {SecToStr(elapsed)}\n')

            # Save results
            df_point.to_pickle(point_estimates_res_file)
            df_est.to_pickle(boot_estimates_res_file)
        else:
            print(f"Loading mixture analysis with {data_label} scores...", flush=True)
            if os.path.isfile(point_estimates_res_file):
                df_point = pd.read_pickle(point_estimates_res_file)
            else:
                warnings.warn(f"Missing data file: {point_estimates_res_file}")
                break
            if os.path.isfile(boot_estimates_res_file):
                df_est = pd.read_pickle(boot_estimates_res_file)
            else:
                warnings.warn(f"Missing data file: {boot_estimates_res_file}")
                break


        # Plot worked examples
        print(f"Plotting application with {data_label} scores...", flush=True)
        fig_ex = plt.figure(figsize=(12, 4))
        gs = plt.GridSpec(nrows=1, ncols=2, hspace=0.15, wspace=0.15,
                          left=0.10, right=0.95, bottom=0.12, top=0.96)

        # sns.set_style("ticks")
        with sns.axes_style("ticks"):
            ax_dists_ex = fig_ex.add_subplot(gs[0, 0])
            plot_distributions(scores, bins, data_label, ax=ax_dists_ex)

        # with sns.axes_style("whitegrid"):
        with sns.axes_style("ticks", {"axes.grid": True, "axes.spines.left": False, 'ytick.left': False}):
            ax_ci_ex = fig_ex.add_subplot(gs[0, 1])
            plot_bootstraps(df_pe, correction=correction, prop_Ref1=prop_Ref1, 
                            ax=ax_ci_ex, limits=application_xlims[data_label], 
                            ci_method=CI_METHOD, initial=False, legend=False, 
                            violins=True, orient='h')
            # if application_xlims[data_label]:
            #     ax_ci_ex.set_xlim(*application_xlims[data_label])

        fig_ex.savefig(os.path.join(fig_dir, f'application_{data_label}.png'))
        fig_ex.savefig(os.path.join(fig_dir, f'application_{data_label}.svg'), transparent=True)


        # Plot selected violins
        print("Plotting violins of constructed mixtures with {} scores...".format(data_label), flush=True)
        plot_mixes = [0]
        for mix in plot_mixes:  #range(n_seeds):
            plot_selected_violins(scores, bins, df_point, df_est, methods, p_stars, sizes,
                                  out_dir, data_label, selected_mix=mix,
                                  add_ci=True, alpha=0.05, ci_method=CI_METHOD,
                                  correction=correction)



        # Plot selected violins
        if False:
            #for mix in range(n_seeds):
            mix = selected_mix
            fig_vio = plt.figure(figsize=(8, 2*len(sizes)))
            ax_vio = fig_vio.add_subplot(111)

            for p, p_star in enumerate(p_stars):
                # df = df_est[np.isclose(p_star, df_est['p1*']) & np.isclose(mix, df_est['Mix'])]
                df = df_est[np.isclose(p_star, df_est['p1*']) & (df_est['Mix'] == mix)]
                # df.sort_values(by='Size', ascending=False, inplace=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    g = sns.violinplot(x='Estimate', y='Size', hue='Method', data=df,
                                       ax=ax_vio, orient='h', cut=0)
                g.axvline(x=p_star, ymin=0, ymax=1, ls='--')  # ,
                          # label="Ground Truth: {:3.2}".format(p_star))
                handles, labels = g.get_legend_handles_labels()
                g.legend(handles, labels[:len(methods)], title="Method")
                # g.legend.set_title("Method")

            sns.despine(top=True, bottom=False, left=False, right=True, trim=True)
            g.invert_yaxis()
            fig_vio.savefig(os.path.join(fig_dir, 'violin_bootstraps_{}_{}.png'.format(mix, data_label)))

        # Plot violin stack
        if False:
            print("Plotting violin stacks with {} scores...".format(data_label), flush=True)
            fig_stack = plt.figure(figsize=(16, 2*len(sizes)))
            gs = plt.GridSpec(nrows=1, ncols=len(methods), hspace=0.2, wspace=0.015)

            for m, method in enumerate(methods):
                ax_stack = fig_stack.add_subplot(gs[m])
                ax_stack.set_title(method)

                if m > 0:
                    plt.setp(ax_stack.get_yticklabels(), visible=False)
                    ax_stack.set_ylabel("")
                    sns.despine(top=True, bottom=False, left=True, right=True, trim=True)
                else:
                    sns.despine(top=True, bottom=False, left=False, right=True, trim=True)

                ax_stack.set_xlim((0, 1))
                for p, p_star in enumerate(p_stars):
                    df = df_est[np.isclose(p_star, df_est['p1*'])]  # & np.isclose(size, df_vio['Size'])]
                    # df.sort_values(by='Size', ascending=False, inplace=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        g = sns.violinplot(x='Estimate', y='Size', hue='Mix', data=df[df['Method'] == method],
                                           ax=ax_stack, orient='h', cut=0)
                    g.axvline(x=p_star, ymin=0, ymax=1, ls='--')  # ,
                               #label="Ground Truth: {:3.2}".format(p_star))
                    handles, labels = g.get_legend_handles_labels()
                    g.legend(handles, labels[:n_seeds], title="Mixture")
                    # g.legend.set_title("Method")

                g.invert_yaxis()
            fig_stack.savefig(os.path.join(fig_dir, 'violin_stacks_{}.png'.format(data_label)))



        # Plot error bars
        if False:
            orient = 'h'
            c = sns.color_palette()[-3]  # 'gray'
            # fig_err = plt.figure(figsize=(16, 16))
            # ax_ci = fig_err.add_subplot(111)
            for s, size in enumerate(sizes):
                for p, p_star in enumerate(p_stars):
                    df = df_est[(df_est.Size == size) & np.isclose(df_est['p1*'], p_star)]
                    df_means = df.groupby('Method').mean()
                    # Add confidence intervals
                    errors = np.zeros(shape=(2, len(methods)))
                    means = []
                    for midx, method in enumerate(methods):
                        nobs = size  # len(df_est[method])
                        mean_est = df_means.loc[method, 'Estimate']
                        means.append(mean_est)
                        # mean_est = np.mean(df_est[method])
                        count = int(mean_est*nobs)
                        ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method=CI_METHOD)
                        errors[0, midx] = mean_est - ci_low1  # y[midx] - ci_low1
                        errors[1, midx] = ci_upp1 - mean_est  # ci_upp1 - y[midx]

                    # Add white border around error bars
                    # ax.errorbar(x=x, y=y, yerr=errors, fmt='none', c='w', lw=6, capsize=14, capthick=6)
                    if orient == 'v':
                        x = ax_vio.get_xticks()
                        y = means
                        ax_vio.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=7, c=c, lw=4, capsize=12, capthick=4, label="Confidence Intervals ({:3.1%})".format(1-alpha))
                    elif orient == 'h':
                        x = means
                        y = ax_vio.get_yticks()
                        ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='s', markersize=7, c=c, lw=4, capsize=12, capthick=4, label="Confidence Intervals ({:3.1%})".format(1-alpha))


                    g.axvline(x=p_star, ymin=0, ymax=1, ls='--')  # ,
                              # label="Ground Truth: {:3.2}".format(p_star))
                    handles, labels = g.get_legend_handles_labels()
                    g.legend(handles, labels[:len(methods)], title="Method")

        #     sns.despine(top=False, bottom=True, left=True, right=False, trim=True)
            sns.despine(top=True, bottom=False, left=False, right=True, trim=True)
            g.invert_yaxis()
            fig_vio.savefig(os.path.join(fig_dir, 'ci_selection_{}.png'.format(data_label)))

        # Plot mixture distributions - add reference populations?
        if False:
            print("Plotting constructed mixtures from {} scores...".format(data_label), flush=True)
            fig_mixes = plt.figure(figsize=(8, 8))
            gs = plt.GridSpec(nrows=len(sizes), ncols=1, hspace=0.15, wspace=0.15)
            for si, size in enumerate(sizes):
                mix_dist_file = '{}/mix{}_size{}_{}.pkl'.format(out_dir, selected_mix, size, data_label)
                if os.path.isfile(mix_dist_file):
                    df_mixes = pd.read_pickle(mix_dist_file)
                    if si == 0:
                        ax_mixes = fig_mixes.add_subplot(gs[-(si+1)])
                        ax_base = ax_mixes
                    else:
                        ax_mixes = fig_mixes.add_subplot(gs[-(si+1)], sharex=ax_base)
                        plt.setp(ax_mixes.get_xticklabels(), visible=False)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        sns.distplot(scores["Ref1"], bins=bins['edges'], hist=False, kde=True, kde_kws={"shade": True}) # hist=True, norm_hist=True, kde=False)#,
                                     # label=r"$R_1$", ax=ax_mixes)
                        for p, p_star in enumerate(p_stars):
                            sns.distplot(df_mixes[p_star], bins=bins['edges'], hist=False,
                                         label=r"$p_1^*={}$".format(p_star), ax=ax_mixes)  # \tilde{{M}}: n={},
                        sns.distplot(scores["Ref2"], bins=bins['edges'], hist=False, kde=True, kde_kws={"shade": True})#,
                                     # label=r"$R_2$", ax=ax_mixes)

                    ax_mixes.set_ylabel(r"$n={}$".format(size), rotation='horizontal')
                    ax_mixes.set_xlabel("")
                    # Remove y axis
                    sns.despine(top=True, bottom=False, left=True, right=True, trim=True)
                    ax_mixes.set_yticks([])
                    ax_mixes.set_yticklabels([])
                    if si == len(p_stars)-1:
                        ax_mixes.legend()  # title=r"$p_1^*$") #, ncol=len(p_stars))
                    else:
                        ax_mixes.get_legend().set_visible(False)
                else:
                    warnings.warn("Missing data file: {}".format(mix_dist_file))
                    break
            # ax_base.set_xlabel("GRS")
            ax_base.set_xlabel("Genetic Risk Score")
            fig_mixes.savefig(os.path.join(fig_dir, 'constructions_{}.png'.format(data_label)))


        # Plot distributions around the estimated proportion with given sample_size from the characterisation data
        # if prop_Ref1 is not None:
        #     errors[]
        # if prop_Ref1 is None:
        #     prop_Ref1 = np.mean()

        test_bootstrap_convergence = False
        if test_bootstrap_convergence:
            print("Plotting bootstrap convergence with {} scores...".format(data_label), flush=True)
            # Explore the effect of n_boots
            n_boots = [0, 1, 10, 100, 1000]

            fig, axes = plt.subplots(len(n_boots), 1, sharex=True, figsize=(9, 2*len(n_boots)))

            for b, n_boots in enumerate(n_boots):
                if n_boots == 0:
                    df_bs = df_pe.iloc[[0]]
                    df_point = df_pe.iloc[[0]]
                else:
                    df_point = df_pe.iloc[[0]]
                    df_bs = df_pe.iloc[1:n_boots+1, :]

                if b == 0:
                    legend = True
                else:
                    legend = False

                plot_bootstraps(df_pe, correction, prop_Ref1, axes[b], limits=(0, 1), ci_method=CI_METHOD, alpha=alpha, legend=legend, orient='h')

            fig.savefig(os.path.join(fig_dir, "boot_size_{}.png".format(data_label)))

            for mix in range(n_seeds):
                df_mix = df_est[(df_est['Mix'] == mix)]
                frames = []

                for b in [1000, 100, 10, 1]:
                    for s, size in enumerate(sizes):
                        for p, p_star in enumerate(p_stars):
                            df_ps = df_mix[np.isclose(p_star, df_mix['p1*'])
                                           & (df_mix['Size'] == size)]

                            for m, method in enumerate(methods):
                                df_meth = df_ps[df_ps["Method"] == method]
                                df_tmp = df_meth[:b]
                                df_tmp["Error"] = df_tmp["Estimate"] - p_star
                                df_tmp["Bootstraps"] = b
                                frames.append(df_tmp)

                                print("Bootstraps = {}; Method: {}; p_C = {}; size = {}; Mean = {}; SD = {}"
                                      .format(b, method, p_star, size, np.mean(df_tmp["Estimate"] - p_star), np.std(df_tmp["Estimate"] - p_star)))
                                #.sample(n=100, replace=False)

                df = pd.concat(frames, ignore_index=True)

                with sns.axes_style("whitegrid"):
                    g = sns.catplot(x="Bootstraps", y="Error", hue="Method", col="p1*", row="Size", row_order=sizes[::-1], data=df, kind="violin") #, margin_titles=True) #, ax=ax_err)
                    g.set(ylim=(-0.5, 0.5)).despine(left="True")
                    # g.despine(left="True")

                plt.savefig(os.path.join(fig_dir, "bootstraps_{}_{}.png".format(mix, data_label)))
