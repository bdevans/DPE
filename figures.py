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
import scipy as sp
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import tqdm

import proportion_estimation as pe
from datasets import (load_diabetes_data, load_renal_data)


# ---------------------------- Define constants ------------------------------

# METHODS_ORDER = ["Excess", "Means", "EMD", "KDE"]
FRESH_DATA = False  # CAUTION!
#out_dir = "results_test"
# out_dir = "results_100_100"
# out_dir = "results/m1000_b0"
out_dir = "results/m1000_b10"
fig_dir = os.path.join(out_dir, "figs")

verbose = False


seed = 42
n_boot = 1000
n_mix = 5 #100
sample_size = 1000  # -1
n_mixes = 5
selected_mix = 0
alpha = 0.05
CI_METHOD = "experimental"  # "stderr" # "centile" "jeffreys"
# normal : asymptotic normal approximation
# agresti_coull : Agresti-Coull interval
# beta : Clopper-Pearson interval based on Beta distribution
# wilson : Wilson Score interval
# jeffreys : Jeffreys Bayesian Interval
# binom_test : experimental, inversion of binom_test
# http://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html

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
mpl.rc('axes', titlesize=14)     # fontsize of the axes title
mpl.rc('axes', labelsize=14)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=12)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=12)    # fontsize of the tick labels
mpl.rc('legend', fontsize=11)    # legend fontsize
mpl.rc('figure', titlesize=14)  # fontsize of the figure title
mpl.rc('lines', linewidth=2)
mpl.rc('figure', dpi=100)
mpl.rc('savefig', dpi=300)


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


def get_error_bars(df, average=np.mean, alpha=0.05, ci_method="stderr"):
    """df: columns are method names"""

    #methods = list(df.columns)
    #n_est, n_methods = df.shape
    n_methods = len(df.columns)
    errors = np.zeros(shape=(2, n_methods))

    for m, method in enumerate(df):  # enumerate(methods):

        ci_low, ci_upp = pe.calc_conf_intervals(df[method].values,
                                                average=average, alpha=alpha,
                                                ci_method=ci_method)
        errors[0, m] = means[m] - ci_low
        errors[1, m] = ci_upp - means[m]

    return errors


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
            CS = ax.contourf(proportions, sample_sizes, average_error,
                             SHADING_LEVELS, cmap=cmap, extend='both')  # , vmin=-.05, vmax=.05)
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
    gs = plt.GridSpec(nrows=2, ncols=4, hspace=0.15, wspace=0.15)

    for m, method in enumerate(pe._ALL_METHODS_):  # enumerate(methods):

        # Plot average accuracy across mixtures
        ax_acc = fig.add_subplot(gs[0, m], xticklabels=[])
        if m == 0:
            ax_acc.set_ylabel('Sample size ($n$)')
        else:
            ax_acc.set_yticklabels([])
        plot_accuracy(estimates, proportions, sample_sizes, method, fig, ax_acc, contour_levels=cl)

        # Plot deviation across mixtures
        ax_dev = fig.add_subplot(gs[1, m])
        if m == 0:
            ax_dev.set_ylabel('Sample size ($n$)')
        else:
            ax_dev.set_yticklabels([])
        ax_dev.set_xlabel('$p_D^*$')  # '$p_1^*$'
        plot_deviation(estimates, proportions, sample_sizes, method, fig, ax_dev, title=False)

    return fig


def plot_distributions(scores, bins, data_label, ax=None):

    if not ax:
        f, ax = plt.subplots()

    palette = ['#6f92f3', '#aac7fd', '#bbbbbb', '#f7b89c', '#e7745b']
    # TODO: Melt data together so the y-axis is normalised properly?

    with sns.axes_style("ticks") and warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sns.distplot(scores['Ref1'], bins=bins['edges'], norm_hist=False,
                     label="$R_C: n={:,}$".format(len(scores['Ref1'])),
                     # label="$R_1: n={:,}$".format(len(scores['Ref1'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[-1])
        sns.distplot(scores['Ref2'], bins=bins['edges'], norm_hist=False,
                     label="$R_N: n={:,}$".format(len(scores['Ref2'])),
                     # label="$R_2: n={:,}$".format(len(scores['Ref2'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[0])
        sns.distplot(scores['Mix'], bins=bins['edges'], norm_hist=False,
                     label=r"$\tilde{{M}}: n={:,}$".format(len(scores['Mix'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[2])

        sns.despine(top=True, bottom=False, left=False, right=True, trim=True)

    #ax.yaxis.tick_left()
    #ax.yaxis.set_ticks_position('left')

    # if len(indep_vars) > 1:
    #     Use jointplot
    ax.set_xlabel("GRS")
    ax.legend()
    # plt.savefig('figs/distributions_{}.png'.format(data_label))


def plot_bootstraps(df_bs, df_point=None, prop_Ref1=None, ax=None, limits=None, alpha=0.05,
                    ci_method='jeffreys', legend=True, orient='v'):

    c = sns.color_palette()[-3]  # 'gray'

    # df_bs = df_bs[pe._ALL_METHODS_]

    if not ax:
        f, ax = plt.subplots()
    if limits:
        if orient == 'v':
            ax.set_ylim(limits)
        if orient == 'h':
            ax.set_xlim(limits)

    # Draw violin plots of bootstraps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sns.violinplot(data=df_bs, orient=orient, ax=ax, cut=0)  # , inner="stick")
    if orient == 'v':
        sns.despine(ax=ax, top=True, bottom=True, left=False, right=True, trim=True)
    elif orient == 'h':
        sns.despine(ax=ax, top=True, bottom=False, left=False, right=True, trim=True)

    if prop_Ref1:  # Add ground truth
        truth_label = r"$\tilde{{p}}_C: {:4.3}$".format(prop_Ref1)
        if orient == 'v':
            ax.axhline(y=prop_Ref1, xmin=0, xmax=1, ls='--',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(prop_Ref1))
        elif orient == 'h':
            ax.axvline(x=prop_Ref1, ymin=0, ymax=1, ls='--',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(prop_Ref1))

    # Add confidence intervals # TODO: Refactor
#    if orient == 'v':
#        x, y = ax.get_xticks(), df_bs.mean().values
#        means = y
#    elif orient =='h':
#        x, y = df_bs.mean().values, ax.get_yticks()
#        means = x

    if orient == 'v':
        x, y = ax.get_xticks(), df_point.iloc[0].values
        means = y
    elif orient =='h':
        x, y = df_point.iloc[0].values, ax.get_yticks()
        means = x

    # TODO: Think about means
    # get_error_bars(df, average=np.mean, alpha=0.05, ci_method="stderr")
    errors = np.zeros(shape=(2, len(methods)))

    for midx, method in enumerate(df_bs):  # enumerate(methods):

        initial = df_point.iloc[0][method]

        ci_low, ci_upp = pe.calc_conf_intervals(df_bs[method], initial=initial, average=np.mean, alpha=alpha, ci_method=ci_method)

#        if ci_method == 'centile':
#            ci_low, ci_upp = np.percentile(df_bs[method], [100*alpha/2, 100-(100*alpha/2)])
#        elif ci_method == 'stderr':
#            p = np.mean(df_bs[method])
#            nobs = len(df_bs[method])
#            # err = np.sqrt(p*(1-p)/nobs) * 1.96  # TODO: Lookup this value for a given alpha
#            err = np.sqrt(p*(1-p)/nobs) * sp.stats.norm.ppf(alpha/2)
#            ci_low, ci_upp = p - err, p + err
#        else:
#            nobs = len(df_bs[method])
#            count = int(np.mean(df_bs[method])*nobs)
#            ci_low, ci_upp = proportion_confint(count, nobs, alpha=alpha,
#                                                method=ci_method)
        errors[0, midx] = means[midx] - ci_low
        errors[1, midx] = ci_upp - means[midx]

    # Add white border around error bars
    # ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c='w', lw=8, capsize=12, capthick=8)

#    error_label = "Confidence Intervals ({:3.1%})".format(1-alpha)
    error_label = "{:3.1%} CI".format(1-alpha)
    if orient == 'v':
        ax.errorbar(x=x, y=y, yerr=errors, fmt='*', markersize=9, c=c, lw=5,
                    capsize=12, capthick=3, label=error_label)
    elif orient == 'h':
        ax.errorbar(x=x, y=y, xerr=errors, fmt='*', markersize=9, c=c, lw=5,
                    capsize=12, capthick=3, label=error_label)

    if orient == 'v':
        ax.yaxis.tick_left()
        ax.set_ylabel("$p_C$", {"rotation": "horizontal"})  # "$p_1$"
        # ax.set_xticks([])  # Remove ticks for method labels
    elif orient == 'h':
        ax.xaxis.tick_bottom()
        ax.set_xlabel("$p_C$")  # "$p_1$"
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


def plot_selected_violins(scores, bins, df_point, df_boots, methods, p_stars, sizes, out_dir, data_label, selected_mix=0, add_ci=True, alpha=0.05, ci_method="jeffreys"):

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
    gs = plt.GridSpec(nrows=len(sizes), ncols=2, width_ratios=[3, 2], hspace=0.2, wspace=0.005)

    for si, size in enumerate(sizes):
#        ax_vio = fig_select.add_subplot(gs[-(si+1), :-1])
#        ax_mix = fig_select.add_subplot(gs[-(si+1), -1])
        mix_dist_file = '{}/mix{}_size{}_{}.pkl'.format(out_dir, selected_mix, size, data_label)
        if not os.path.isfile(mix_dist_file):
            warnings.warn("File not found: {}".format(mix_dist_file))
            return
        df_mixes = pd.read_pickle(mix_dist_file)

        if si == 0:
            # Save base axes
            ax_vio = fig_select.add_subplot(gs[-(si+1), :-1])
            ax_vio_base = ax_vio
#            sns.plt.xlim(0, 1)
#            vio_xlabs = ax_vio_base.get_xticklabels()
            ax_mix = fig_select.add_subplot(gs[-(si+1), -1])
            ax_mix_base = ax_mix
        else:
            ax_vio = fig_select.add_subplot(gs[-(si+1), :-1], sharex=ax_vio_base)
            plt.setp(ax_vio.get_xticklabels(), visible=False)
            ax_mix = fig_select.add_subplot(gs[-(si+1), -1], sharex=ax_mix_base)
            plt.setp(ax_mix.get_xticklabels(), visible=False)

        # Plot constructed mixture distributions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sns.distplot(scores["Ref2"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True}, # hist=True, norm_hist=True, kde=False)#,
                         label=r"$p_C^*=0.0\ (R_N)$", ax=ax_mix, color=palette[0])
                         # label=r"$p_1^*=0.0\ (R_2)$", ax=ax_mix, color=palette[0])
            for p, p_star in enumerate(p_stars):
                sns.distplot(df_mixes[p_star], bins=bins['edges'], hist=False,
                             label=r"$p_C^*={}$".format(p_star), ax=ax_mix, color=palette[p+1])  # \tilde{{M}}: n={},
                             # label=r"$p_1^*={}$".format(p_star), ax=ax_mix, color=palette[p+1])  # \tilde{{M}}: n={},
            sns.distplot(scores["Ref1"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True},
                         label=r"$p_C^*=1.0\ (R_C)$", ax=ax_mix, color=palette[len(p_stars)+1])
                         # label=r"$p_1^*=1.0\ (R_1)$", ax=ax_mix, color=palette[len(p_stars)+1])

        # Plot violins of bootstrapped estimates
        for p, p_star in enumerate(p_stars):

            # Add annotations for p1*
            ax_vio.axvline(x=p_star, ymin=0, ymax=1, ls='--', lw=3, zorder=0, color=palette[p+1])
                       #label="Ground Truth: {:3.2}".format(p_star))

            # Add shading around the true values
            shade_span = 0.02
            ax_vio.axvspan(p_star-shade_span, p_star+shade_span, alpha=0.5, zorder=0, color=palette[p+1])


            # Select estimates at p_star and size for all methods
            df = df_boots[np.isclose(p_star, df_est['p1*']) &
                          (df_est['Size'] == size) &
                          (df_est["Mix"] == selected_mix)]
#                          np.isclose(size, df_est['Size']) &
#                          np.isclose(selected_mix, df_est["Mix"])]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
    #            g = sns.violinplot(x='Estimate', y='Size', hue='Method', data=df, ax=ax_vio, orient='h', cut=0, linewidth=2)
                sns.violinplot(x='Estimate', y='Method', data=df, ax=ax_vio, orient='h', cut=0, linewidth=2, color=palette[p+1])

#            handles, labels = g.get_legend_handles_labels()
#            g.legend(handles, labels[:len(methods)], title="Method")
            ax_vio.set_ylabel(r"$n={:,}$".format(size))  # , rotation='horizontal')
            ax_vio.set_xlabel("")
#            ax_vio.set_xticklabels([])
            # Remove y axis
#            sns.despine(ax=ax_vio, top=True, bottom=False, left=True, right=True, trim=True)
#            ax_vio.set_yticks([])
#            ax_vio.set_yticklabels([])
#            ax_vio.get_yticklabels().set_visible(False)
            ax_vio.set(xlim=(0, 1))

            if add_ci:  # Add confidence intervals
                # The true value will be within these bars for 95% of samples (not measures)
                # For alpha = 0.05, the CI bounds are +/- 1.96*SEM
                df_means = df.groupby('Method').mean()
                errors = np.zeros(shape=(2, len(methods)))
                means = []
                initials = []
                for midx, method in enumerate(pe._ALL_METHODS_):  # enumerate(methods):

                    mean_est = df_means.loc[method, 'Estimate']
#                    print(df_point)
#                    initial = df_point.iloc[0][method]
                    initial = df_point[np.isclose(p_star, df_point['p1*']) & (df_point['Size'] == size) & (df_point["Mix"] == selected_mix)][method].values[0]

                    df_p = df.pivot_table(values="Estimate", index=["p1*", "Size", "Mix", "Boot"], columns="Method")

#                    print(df_p, flush=True)
#                    print(df_point, flush=True)
                    ci_low, ci_upp = pe.calc_conf_intervals(df_p[method], initial=initial, average=np.mean, alpha=0.05, ci_method=CI_METHOD)
#                    print(df_point.iloc[0][method])
#                    print(ci_low, ci_upp, flush=True)

                    ### PREVIOUS CI METHOD ###

#                    if ci_method == 'centile':
#                        df_pc = df[df["Method"] == method].loc[:, "Estimate"]
#                        ci_low, ci_upp = np.percentile(df_pc, [100*alpha/2, 100-(100*alpha/2)])
#                    elif ci_method == 'stderr':
#                        # p = np.mean(df_pc)
#                        # nobs = len(df_pc)
#                        p = mean_est
#                        nobs = size
#                        # err = np.sqrt(p*(1-p)/nobs) * 1.96  # TODO: Look up correct value for alpha
#                        err = np.sqrt(p*(1-p)/nobs) * sp.stats.norm.ppf(alpha/2)
#                        ci_low, ci_upp = p - err, p + err
#                    else:  # Assumes Binomial Distribution
#                        nobs = size  # len(df_est[method])
#
#                        # mean_est = np.mean(df_est[method])
#                        count = int(mean_est*nobs)
#                        ci_low, ci_upp = proportion_confint(count, nobs, alpha=alpha, method=ci_method)
#                        # ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method=ci_method)
#                        # ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')# ax.plot(x, y, marker='o', ls='', markersize=20)


                    means.append(mean_est)
                    initials.append(initial)
#                    errors[0, midx] = mean_est - ci_low  # y[midx] - ci_low1
#                    errors[1, midx] = ci_upp - mean_est  # ci_upp1 - y[midx]
                    errors[0, midx] = initial - ci_low  # y[midx] - ci_low1
                    errors[1, midx] = ci_upp - initial  # ci_upp1 - y[midx]
                # print(errors)

#                x = means
                x = initials
                y = ax_vio.get_yticks()

                # Add white border around error bars
                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='none', c=(0.45, 0.45, 0.45), lw=5, capsize=14, capthick=5)

                ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='*', markersize=18 ,
                                c=palette[p+1], lw=2, capsize=12, capthick=2,
                                label="Confidence Intervals ({:3.1%})".format(1-alpha),
                                markeredgecolor=(0.45, 0.45, 0.45))

        # sns.despine(top=False, bottom=True, left=True, right=False, trim=True)
#        if si == 0:
#            ax_mix_base.legend() #title=r"$p_1^*$") #, ncol=len(p_stars))

        if si == len(sizes)-1:  # Top row
#            sns.despine(ax=ax_vio, top=True, bottom=False, left=True, right=True, trim=True)
#            sns.despine(ax=ax_mix, top=True, bottom=False, left=True, right=True, trim=True)
#            ax_vio_base.legend(title="Method")
            ax_mix.legend() #title=r"$p_1^*$") #, ncol=len(p_stars))
#            ax_vio.set_xticklabels(vio_xlabs)
        else:
#            sns.despine(ax=ax_vio, top=True, bottom=True, left=True, right=True, trim=True)
#            sns.despine(ax=ax_mix, top=True, bottom=True, left=True, right=True, trim=True)
#            ax_vio.get_legend().set_visible(False)
#            pass
#            if si != 0:
#                ax_vio.set_xticklabels([])
#                ax_mix.set_xticklabels([])
#            ax_vio.set_xlabel("")
#            plt.setp(ax_vio.get_xticklabels(), visible=False)
            # Remove legend and label yaxis instead
            ax_mix.get_legend().set_visible(False)



#        ax_vio.set_ylabel(r"$n={}$".format(size), rotation='horizontal')
#        ax_vio.set_xlabel("")
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
#    g.invert_yaxis()
    ax_vio_base.set_xlabel("Estimated prevalence ($p_C$)")  # $p_1$
    ax_mix_base.set_xlabel("GRS")
#    ax_vio_base.set_xticks()
#    plt.setp(ax_vio.get_xticklabels(), visible=True)
#    plt.tight_layout()
    fig_select.savefig(os.path.join(fig_dir, 'violin_selection_{}_{}.png'.format(selected_mix, data_label)))


def plot_compound_figure():
    """No longer used"""
    fig = plt.figure(figsize=(12,8))
    gs = plt.GridSpec(nrows=2, ncols=3, hspace=0.15, wspace=0.15)
    estimates = boots_estimates

    #sns.set_style("ticks")
    ax_dists = fig.add_subplot(gs[-1,-1])
    # with sns.axes_style("ticks"):
    plot_distributions(scores, bins, data_label, ax=ax_dists)

    #sns.set_style("whitegrid")
    ax_ci = fig.add_subplot(gs[0,-1])
    # with sns.axes_style("whitegrid"):
        #plot_bootstraps(df_pe, prop_Ref1, ax_ci, ylims=(0, 0.12))
    plot_bootstraps(df_pe, prop_Ref1, ax_ci, limits=None, ci_method=CI_METHOD)


    # Plot average accuracy across bootstraps
    cl = [0.02]
    ax_grid_Means = fig.add_subplot(gs[0,0], xticklabels=[])
    ax_grid_Means.set_ylabel('Sample size ($n$)')
#    ax_p2 = ax_grid_Means.twiny()
#    ax_p2.set_xticklabels(ax_grid_Means.get_xticks()[::-1])
    plot_accuracy(estimates, proportions, sample_sizes, "Means", fig, ax_grid_Means, contour_levels=cl)

    ax_grid_Excess = fig.add_subplot(gs[0,1], xticklabels=[], yticklabels=[])
    plot_accuracy(estimates, proportions, sample_sizes, "Excess", fig, ax_grid_Excess, contour_levels=cl)

    ax_grid_EMD = fig.add_subplot(gs[1,0])
    ax_grid_EMD.set_ylabel('Sample size ($n$)')
    ax_grid_EMD.set_xlabel('$p_1^*$')
    plot_accuracy(estimates, proportions, sample_sizes, "EMD", fig, ax_grid_EMD, contour_levels=cl)

    ax_grid_KDE = fig.add_subplot(gs[1,1], yticklabels=[])
    ax_grid_KDE.set_xlabel('$p_1^*$')
    plot_accuracy(estimates, proportions, sample_sizes, "KDE", fig, ax_grid_KDE, contour_levels=cl)

    #axes = np.array([[ax_grid_Means, ax_grid_Excess], [ax_grid_EMD, ax_grid_KDE]])
    #plot_bootstrap_results(out_dir, data_label, fig, axes)  # "T1GRS"

#    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'compound_{}.png'.format(data_label)))


    # Plot deviation across bootstraps
    fig_dev = plt.figure(figsize=(8,8))
    gs = plt.GridSpec(nrows=2, ncols=2, hspace=0.15, wspace=0.15)

    ax_dev_Means = fig_dev.add_subplot(gs[0,0], xticklabels=[])
    ax_dev_Means.set_ylabel('Sample size ($n$)')
    plot_deviation(estimates, proportions, sample_sizes, "Means", fig_dev, ax_dev_Means)

    ax_dev_Excess = fig_dev.add_subplot(gs[0,1], xticklabels=[], yticklabels=[])
    plot_deviation(estimates, proportions, sample_sizes, "Excess", fig_dev, ax_dev_Excess)

    ax_dev_EMD = fig_dev.add_subplot(gs[1,0])
    ax_dev_EMD.set_ylabel('Sample size ($n$)')
    ax_dev_EMD.set_xlabel('$p_1^*$')
    plot_deviation(estimates, proportions, sample_sizes, "EMD", fig_dev, ax_dev_EMD)

    ax_dev_KDE = fig_dev.add_subplot(gs[1,1], yticklabels=[])
    ax_dev_KDE.set_xlabel('$p_1^*$')
    plot_deviation(estimates, proportions, sample_sizes, "KDE", fig_dev, ax_dev_KDE)

#    fig_dev.tight_layout()
    fig_dev.savefig(os.path.join(fig_dir, 'deviation_{}.png'.format(data_label)))




plot_results = False

if plot_results:
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


    for data_label, data in [("Diabetes", load_diabetes_data('T1GRS')),
                             ("Renal", load_renal_data())]:

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
                                       alpha=alpha, true_p1=prop_Ref1, n_jobs=-1,
                                       logfile="{}/pe_{}.log".format(out_dir, data_label))

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

        # Plot bootstrapped estimates of p1
        if bool(boots_estimates):
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
        point_estimates_res_file = '{}/pe_stack_analysis_point_{}.pkl'.format(out_dir, data_label)
        boot_estimates_res_file = '{}/pe_stack_analysis_{}.pkl'.format(out_dir, data_label)
        if FRESH_DATA:
            print("Running mixture analysis with {} scores...".format(data_label), flush=True)
            t = time.time()  # Start timer

            violin_scores = {}
            violin_scores['Ref1'] = scores['Ref1']
            violin_scores['Ref2'] = scores['Ref2']
            dfs_point = []
            dfs_boot = []

            size_bar = tqdm.tqdm(sizes, dynamic_ncols=True)
            for s, size in enumerate(size_bar):
                size_bar.set_description("Size = {:6,}".format(size))
                Mixtures = {mix: {} for mix in range(n_mixes)}

                for mix in tqdm.trange(n_mixes, dynamic_ncols=True, desc=" Mix"):
                    mix_dist_file = '{}/mix{}_size{}_{}.pkl'.format(out_dir, mix, size, data_label)

                    prop_bar = tqdm.tqdm(p_stars, dynamic_ncols=True)
                    for p, p_star in enumerate(prop_bar):
                        prop_bar.set_description(" p1* = {:6.2f}".format(p_star))

                        violin_scores['Mix'] = construct_mixture(scores['Ref1'], scores['Ref2'], p_star, size)
                        Mixtures[mix][p_star] = violin_scores['Mix']
                        df_cm = pe.analyse_mixture(violin_scores, bins, methods,
                                                   n_boot=n_boot, boot_size=size,
                                                   n_mix=n_mix,
                                                   alpha=alpha, true_p1=p_star,
                                                   n_jobs=-1, verbose=0)
                        df_point = df_cm.iloc[[0]]
#                        print(type(df_point), flush=True)
#                        print(df_point, flush=True)
                        df_point['Size'] = size
                        df_point['p1*'] = p_star
                        df_point['Mix'] = mix
#                        df_point = df_point.melt(var_name='Method', id_vars=['p1*', 'Size', 'Mix'], value_name='Estimate')
                        dfs_point.append(df_point)
#                        print(type(df_point), flush=True)
#                        print(df_point, flush=True)

                        df_boots = df_cm.iloc[1:, :]
                        if n_mix > 0:
                            n_bootstraps = n_mix * n_boot
                        else:
                            n_bootstraps = n_boot
                        df_boots['Size'] = size * np.ones(n_bootstraps, dtype=int)
                        df_boots['p1*'] = p_star * np.ones(n_bootstraps)
                        df_boots['Mix'] = mix * np.ones(n_bootstraps, dtype=int)
                        df_boots["Boot"] = list(range(n_bootstraps))
                        df_boots = df_boots.melt(var_name='Method', id_vars=['p1*', 'Size', 'Mix', "Boot"], value_name='Estimate')
                        dfs_boot.append(df_boots)

                    df_size = pd.DataFrame(Mixtures[mix], columns=p_stars)
                    df_size.to_pickle(mix_dist_file)

            df_point = pd.concat(dfs_point, ignore_index=True)
            df_est = pd.concat(dfs_boot, ignore_index=True)
            elapsed = time.time() - t
            print('Elapsed time = {}\n'.format(SecToStr(elapsed)))

            # Save results
            df_point.to_pickle(point_estimates_res_file)
            df_est.to_pickle(boot_estimates_res_file)
        else:
            print("Loading mixture analysis with {} scores...".format(data_label), flush=True)
            if os.path.isfile(point_estimates_res_file):
                df_point = pd.read_pickle(point_estimates_res_file)
            else:
                warnings.warn("Missing data file: {}".format(point_estimates_res_file))
                break
            if os.path.isfile(boot_estimates_res_file):
                df_est = pd.read_pickle(boot_estimates_res_file)
            else:
                warnings.warn("Missing data file: {}".format(boot_estimates_res_file))
                break



        # Plot selected violins
        if False:
            #for mix in range(n_mixes):
            mix = selected_mix
            fig_vio = plt.figure(figsize=(8, 2*len(sizes)))
            ax_vio = fig_vio.add_subplot(111)

            for p, p_star in enumerate(p_stars):
#                df = df_est[np.isclose(p_star, df_est['p1*']) & np.isclose(mix, df_est['Mix'])]
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
            # gs = plt.GridSpec(nrows=len(sizes), ncols=len(p_stars), hspace=0.15, wspace=0.15)
            # gs = plt.GridSpec(nrows=len(sizes), ncols=1, hspace=0.15, wspace=0.15)
        #    ax_vio = fig_vio.add_subplot(111)
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
                    g.legend(handles, labels[:n_mixes], title="Mixture")
                    # g.legend.set_title("Method")

        #        sns.despine(top=True, bottom=False, left=False, right=True, trim=True)
                g.invert_yaxis()
            fig_stack.savefig(os.path.join(fig_dir, 'violin_stacks_{}.png'.format(data_label)))


        # Plot selected violins
        print("Plotting violins of constructed mixtures with {} scores...".format(data_label), flush=True)
        plot_mixes = [0]
        for mix in plot_mixes:  #range(n_mixes):
            plot_selected_violins(scores, bins, df_point, df_est, methods, p_stars, sizes,
                                  out_dir, data_label, selected_mix=mix,
                                  add_ci=True, alpha=0.05, ci_method=CI_METHOD)


        # Plot error bars
        if False:
            orient = 'h'
            c = sns.color_palette()[-3]  # 'gray'
            # fig_err = plt.figure(figsize=(16, 16))
            # ax_ci = fig_err.add_subplot(111)
            for s, size in enumerate(sizes):
                for p, p_star in enumerate(p_stars):
                    # df = df_est[np.isclose(p_star, df_est['p1*'])]
#                    df = df_est[np.isclose(df_est.Size, size) & np.isclose(df_est['p1*'], p_star)]
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
                        # ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')# ax.plot(x, y, marker='o', ls='', markersize=20)
                        errors[0, midx] = mean_est - ci_low1  # y[midx] - ci_low1
                        errors[1, midx] = ci_upp1 - mean_est  # ci_upp1 - y[midx]

                    # Add white border around error bars
                    # ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c='w', lw=8, capsize=12, capthick=8)
                    if orient == 'v':
                        x = ax_vio.get_xticks()
                        y = means
                        ax_vio.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=7, c=c, lw=5, capsize=12, capthick=3, label="Confidence Intervals ({:3.1%})".format(1-alpha))
                    elif orient == 'h':
                        x = means
                        y = ax_vio.get_yticks()
                        ax_vio.errorbar(x=x, y=y, xerr=errors, fmt='s', markersize=7, c=c, lw=5, capsize=12, capthick=3, label="Confidence Intervals ({:3.1%})".format(1-alpha))


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
            ax_base.set_xlabel("GRS")
            fig_mixes.savefig(os.path.join(fig_dir, 'constructions_{}.png'.format(data_label)))



        # Plot worked examples
        print("Plotting application with {} scores...".format(data_label), flush=True)
        fig_ex = plt.figure(figsize=(12, 6))
        gs = plt.GridSpec(nrows=1, ncols=2, hspace=0.15, wspace=0.15)

        #sns.set_style("whitegrid")
        ax_ci_ex = fig_ex.add_subplot(gs[0, 0])
        # with sns.axes_style("whitegrid"):
            #plot_bootstraps(df_pe, prop_Ref1, ax_ci, ylims=(0, 0.12))
        plot_bootstraps(df_pe.iloc[1:, :], df_pe.iloc[[0]], prop_Ref1, ax_ci_ex, limits=None, ci_method=CI_METHOD, orient='h')

            #sns.set_style("ticks")
        ax_dists_ex = fig_ex.add_subplot(gs[0, 1])
        with sns.axes_style("ticks"):
            plot_distributions(scores, bins, data_label, ax=ax_dists_ex)

        fig_ex.savefig(os.path.join(fig_dir, 'application_{}.png'.format(data_label)))

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
            # fig_ex = plt.figure(figsize=(6, 2*len(n_boots)))
            # gs = plt.GridSpec(nrows=len(n_boots), ncols=1, hspace=0.15, wspace=0.15, sharex=True)
            fig, axes = plt.subplots(len(n_boots), 1, sharex=True, figsize=(9, 2*len(n_boots)))

            for b, n_boots in enumerate(n_boots):
                # ax_ci_ex = fig_ex.add_subplot(gs[b, 0])
                if n_boots == 0:
                    # df = pe.analyse_mixture(scores, bins, methods, n_boot=0,
                    #                         boot_size=-1, n_mix=n_mix,
                    #                         alpha=alpha, true_p1=prop_Ref1,
                    #                         n_jobs=-1, logfile="{}/pe_{}_{}.log".format(out_dir, n_boots, data_label))
                    df_bs = df_pe.iloc[[0]]
                    df_point = df_pe.iloc[[0]]
                else:
                    # df = df_pe.loc[:n_boots, :]
                    df_point = df_pe.iloc[[0]]
                    df_bs = df_pe.iloc[1:n_boots+1, :]
                # plot_selected_violins(scores, bins, df, methods, p_stars, sizes, out_dir, "b{}_".format(n_boots)+data_label, add_ci=True, alpha=0.05, ci_method="jeffreys")
                if b == 0:
                    legend = True
                else:
                    legend = False
                # plot_bootstraps(df, prop_Ref1, axes[b], limits=(0, 1), ci_method=CI_METHOD, legend=legend, orient='h')
                plot_bootstraps(df_bs, df_point, prop_Ref1, axes[b], limits=(0, 1), ci_method=CI_METHOD, alpha=alpha, legend=legend, orient='h')

            fig.savefig(os.path.join(fig_dir, "boot_size_{}.png".format(data_label)))

            for mix in range(n_mixes):
#                df_mix = df_est[np.isclose(mix, df_est['Mix'])]
                df_mix = df_est[(df_est['Mix'] == mix)]
                frames = []

                for b in [1000, 100, 10, 1]:
                    for s, size in enumerate(sizes):
                        for p, p_star in enumerate(p_stars):
#                            df_ps = df_mix[np.isclose(p_star, df_mix['p1*']) & np.isclose(size, df_mix['Size'])]
                            df_ps = df_mix[np.isclose(p_star, df_mix['p1*']) & (df_mix['Size'] == size)]

                            for m, method in enumerate(methods):
                                df_meth = df_ps[df_ps["Method"] == method]
                                df_tmp = df_meth[:b]
                                df_tmp["Error"] = df_tmp["Estimate"] - p_star
                                df_tmp["Bootstraps"] = b
            #                    df = df.append(df_tmp, ignore_index=True)
                                frames.append(df_tmp)

                                print("Bootstraps = {}; Method: {}; p_C = {}; size = {}; Mean = {}; SD = {}"
                                      .format(b, method, p_star, size, np.mean(df_tmp["Estimate"] - p_star), np.std(df_tmp["Estimate"] - p_star)))
                                #.sample(n=100, replace=False)

                df = pd.concat(frames, ignore_index=True)

                with sns.axes_style("whitegrid"):
                    g = sns.catplot(x="Bootstraps", y="Error", hue="Method", col="p1*", row="Size", row_order=sizes[::-1], data=df, kind="violin") #, margin_titles=True) #, ax=ax_err)
                    g.set(ylim=(-0.5, 0.5)).despine(left="True")

            #        g = sns.catplot(x="Estimate", y="Bootstraps", hue="Method", col="p1*", row="Size", row_order=sizes[::-1], data=df, kind="violin", margin_titles=True) #, ax=ax_err)
                    #g.despine(left="True")

                plt.savefig(os.path.join(fig_dir, "bootstraps_{}_{}.png".format(mix, data_label)))
