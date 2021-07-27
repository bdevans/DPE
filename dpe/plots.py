import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns
from sklearn.metrics import auc

import dpe
from . estimate import calc_conf_intervals, fit_kernel
from . utilities import get_fpr_tpr
# from . utilities import get_roc_scores


def get_error_bars(df_pe, summary=None,
                   scores=None, bins=None, methods=None, 
                   correct_bias=False, average=np.mean, 
                   alpha=0.05, ci_method="bca"):
    """df: columns are method names"""

    #methods = list(df.columns)
    #n_est, n_methods = df.shape
    n_methods = len(df_pe.columns)
    errors = np.zeros(shape=(2, n_methods))
    centres = np.zeros(n_methods)

    # if correct_bias:
    #     centres = correct_estimates(df_pe, average=average)
    for m, method in enumerate(df_pe):  # enumerate(methods):

        boot_values = df_pe.iloc[1:, m]

        if correct_bias:
            # TODO: replace with correct_estimates(df_pe)
            centres[m] = 2 * df_pe.iloc[0, m] - np.mean(boot_values)  # TODO: check np.mean|average
            boot_values = 2 * df_pe.iloc[0, m] - boot_values  # TODO: Check this step
        else:
            centres[m] = df_pe.iloc[0, m]
            # boot_values = df_pe.iloc[1:, m]

        if summary is None or "CI" not in summary[method]:
            # assert scores is not None
            # assert bins is not None
            # assert methods is not None
            ci_low, ci_upp = calc_conf_intervals(boot_values, estimate=centres[m],
                                                scores=scores, bins=bins, 
                                                est_method=methods[method],
                                                correct_bias=False,
                                                average=average, alpha=alpha,
                                                ci_method=ci_method)
        else:
            ci_low, ci_upp = summary[method]["CI"]

        errors[0, m] = centres[m] - ci_low
        errors[1, m] = ci_upp - centres[m]

    return (errors, centres)


def plot_kernels(scores, bins):

    fig, axes = plt.subplots(len(scores), 1, sharex=True)
    X_plot = bins['centers'][:, np.newaxis]
    for (label, data), ax in zip(scores.items(), axes):
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            kde = fit_kernel(data, bins['width'], kernel)
            ax.plot(X_plot[:, 0], np.exp(kde.score_samples(X_plot)), '-',
                    label="kernel = '{0}'; bandwidth = {1}".format(kernel, bins['width']))
        ax.legend(loc='upper left')
        ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
        ax.set_ylabel(label)
    return (fig, axes)


def plot_accuracy(estimates, proportions, sample_sizes, label, fig, ax,
                  shading_levels=None, contour_levels=[0.02], average=np.mean,
                  title=True, cbar=True, linear_colourbar=True, 
                  absolute_error=False):
    # TODO: Plot only one colourbar per row: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html

    if not ax:
        fig, ax = plt.subplots()

    if estimates[label].ndim > 3:  # Take mean over bootstraps first
        average_error = average(np.mean(estimates[label], axis=3), axis=2) - proportions
    else:
        average_error = average(estimates[label], axis=2) - proportions

    if absolute_error:
        average_error = np.abs(average_error)

    # Plot shaded regions for each method on individual subplots
    if not shading_levels:
        if linear_colourbar:
            if absolute_error:
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

    if absolute_error:
        cmap = 'viridis_r'
        colour = [sns.color_palette()[6]]
    else:
        #cmap = "RdBu_r" # mpl.colors.ListedColormap(sns.color_palette("RdBu", n_colors=len(SHADING_LEVELS)))
        #cmap = "bwr"
        cmap = "seismic"
        #cmap = "PuOr"
        colour = [sns.color_palette()[2]]

    if linear_colourbar:
        if absolute_error:
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
                   contour_levels=[0.05], deviation=np.std, 
                   title=True, cbar=True):

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
                          figsize=(16, 8), cl=None, average=np.mean):
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
                    nrows_ncols=(2, len(dpe._ALL_METHODS_)),
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

    for m, method in enumerate(dpe._ALL_METHODS_):  # enumerate(methods):

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
        if m % len(dpe._ALL_METHODS_) == 3:
            cax = grid.cbar_axes[0]
            cax.colorbar(hm, ticks=shading_ticks)  # , extend='both')  # TODO: Fix!
            cax.toggle_label(True)
            cax.axis[cax.orientation].set_label("Average Deviation from $p_C$")

        # Plot deviation across mixtures
        # ax_dev = fig.add_subplot(gs[1, m])
        ax_dev = grid[m+len(dpe._ALL_METHODS_)]
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
        if m % len(dpe._ALL_METHODS_) == 3:
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
        sns.distplot(scores['R_C'], bins=bins['edges'], norm_hist=norm,
                     label="$R_C: n={:,}$".format(len(scores['R_C'])),
                     # label="$R_1: n={:,}$".format(len(scores['R_C'])),
                     ax=ax, kde_kws={'bw': bins['width']}, color=palette[-1])
        sns.distplot(scores['R_N'], bins=bins['edges'], norm_hist=norm,
                     label="$R_N: n={:,}$".format(len(scores['R_N'])),
                     # label="$R_2: n={:,}$".format(len(scores['R_N'])),
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


def plot_bootstraps(df_pe, summary=None,
                    scores=None, bins=None, prepared_methods=None,
                    correct_bias=None, initial=True, p_C=None,
                    ax=None, limits=None, alpha=0.05, ci_method="bca",
                    violins=True, legend=True, orient="v", average=np.mean):

    # c = sns.color_palette()[-3]  # 'gray'
    c = '#999999'
    c_edge = '#777777'

    # df_bs = df_bs[dpe._ALL_METHODS_]

    if not ax:
        f, ax = plt.subplots()
    if limits is None:
        limits = (0, 1)
    else:
        assert isinstance(limits, (tuple, list))
    # if limits:
    #     if orient == 'v':
    #         ax.set_ylim(limits)
    #     if orient == 'h':
    #         ax.set_xlim(limits)

    df_point = df_pe.iloc[0, :]
    df_bs = df_pe.iloc[1:, :]
    methods = df_pe.columns.tolist()
    # TODO: Think...
#    if correct_bias:
#        df_correct = pe.correct_estimates(df_pe)


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

    if p_C:  # Add ground truth
        truth_label = r"$p_C = {:4.3}$".format(p_C)
        if orient == 'v':
            ax.axhline(y=p_C, xmin=0, xmax=1, ls='--', c='#aaaaaa',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(p_C))
        elif orient == 'h':
            ax.axvline(x=p_C, ymin=0, ymax=1, ls='--', c='#aaaaaa',
                       label=truth_label)  # "Ground Truth: {:4.3}".format(p_C))

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

    errors, centres = get_error_bars(df_pe, summary=summary, 
                                    #  scores=scores, bins=bins, methods=prepared_methods,  # TODO: prepared_methods
                                     correct_bias=correct_bias, average=average, 
                                     alpha=alpha, ci_method=ci_method)

    if correct_bias:
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

#    if correct_bias:
#        # Plot initial estimate
#        ax.plot(x, y, 'o', markersize=12, c=(0.25, 0.25, 0.25))
#        ax.plot(x, y, 'o', markersize=8, c=c, label="Initial")

    # errors = np.zeros(shape=(2, len(methods)))

#    for midx, method in enumerate(df_bs):  # enumerate(methods):
#
#        # initial = df_point.iloc[0][method]  # Avoid chained indexing
#        # initial = df_point.iloc[0, midx]
#
#        if correct_bias:
#            # Plot initial estimate
#            ax.plot(x, y, fmt='x', markersize=12, c=(0.25, 0.25, 0.25))
#            ax.plot(x, y, fmt='x', markersize=8, c=c, label="Initial")
#
##            centre = df_correct.iloc[0, midx]
##
##        else:
##            centre = df_point.iloc[0, midx]
##
##        ci_low, ci_upp = calc_conf_intervals(df_bs[method], initial=centre, average=np.mean, alpha=alpha, ci_method=ci_method)
##        errors[0, midx] = means[midx] - ci_low
##        errors[1, midx] = ci_upp - means[midx]



    # Add white border around error bars
    # ax.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=5, c='w', lw=8, capsize=12, capthick=8)

#    error_label = "Confidence Intervals ({:3.1%})".format(1-alpha)
    error_label = f"{1-alpha:3.1%} CI"
#    if correct_bias:
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


    if correct_bias:
        # ax.plot(x, y, '*', markersize=14, c=c, markeredgecolor=c_edge, label="Corrected", zorder=20)
        ax.plot(x, y, 'o', markersize=10, c=c, markeredgecolor=c_edge, label="Corrected", zorder=20)
    else:
        ax.plot(x, y, 'o', markersize=6, c=c, markeredgecolor=c_edge, zorder=20)
        ax.plot(x, y, '.', markersize=2, c=c_edge, markeredgecolor=c_edge, label=r"$\hat{p}_C$", zorder=30)

    if orient == 'v':
        ax.yaxis.tick_left()
        # ax.set_ylabel("$p_C$", {"rotation": "horizontal"})  # "$p_1$"
        ax.set_ylabel("Mixture prevalence", {"rotation": "horizontal"})
        ax.set_ylim(*limits)
        # ax.set_ylim(0, 1)
        # ax.set_xticks([])  # Remove ticks for method labels
    elif orient == 'h':
        ax.xaxis.tick_bottom()
        # ax.set_xlabel("$p_C$")  # "$p_1$"
        ax.set_xlabel("Mixture prevalence")
        ax.set_xlim(*limits)
        # ax.set_xlim(0, 1)
        # ax.set_yticks([])  # Remove ticks for method labels
    #plt.setp(ax, yticks=yticks)
    #ax.yaxis.set_ticks_position('right')

    #ax.set_xticks([])
    #ax.set_xticklabels(list(methods))
    if legend:
        ax.legend()
    # plt.savefig('figs/violins_{}.png'.format(data_label))

    # if False:
    #     # Plot swarm box
    #     f, ax = plt.subplots()
    #     ax = sns.boxplot(data=df_bs)
    #     ax = sns.swarmplot(data=df_bs, color=".25")
    #     sns.despine(top=True, bottom=True, trim=True)
    #     plt.savefig(os.path.join(fig_dir, 'boxes_{}.png'.format(data_label)))


def plot_selected_violins(scores, bins, df_point, df_boots, summaries, #methods, 
                          p_stars, sizes, #out_dir, data_label, 
                          mix_dfs, selected_mix=0,
                          add_ci=True, alpha=0.05, ci_method="bca",
                          correct_bias=False, average=np.mean):  # , fig_dir=""):

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
        print(f"Size = {size} [#{si}/{len(sizes)}]: ", end='', flush=True)
#        ax_vio = fig_select.add_subplot(gs[-(si+1), :-1])
#        ax_mix = fig_select.add_subplot(gs[-(si+1), -1])
        # mix_dist_file = os.path.join(out_dir, f"mix{selected_mix}_size{size}_{data_label}.pkl")
        # mix_dist_file = os.path.join(out_dir, f"ma_{data_label}_size_{size:05d}_mix_{selected_mix:03d}.pkl")
        # if not os.path.isfile(mix_dist_file):
        #     warnings.warn(f"File not found: {mix_dist_file}")
        #     return
        # df_mixes = pd.read_pickle(mix_dist_file)
        df_mixes = mix_dfs[si][selected_mix]

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
            sns.distplot(scores["R_N"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True}, # hist=True, norm_hist=True, kde=False)#,
                         label=r"$p_C=0.0\ (R_N)$", ax=ax_mix, color=palette[0])
            for p, p_star in enumerate(p_stars):
                sns.distplot(df_mixes[p_star], bins=bins['edges'], hist=False,
                             label=r"$p_C={}$".format(p_star), ax=ax_mix, color=palette[p+1])  # \tilde{{M}}: n={},
            sns.distplot(scores["R_C"], bins=bins['edges'], hist=False, kde=True,
                         kde_kws={"shade": True},
                         label=r"$p_C=1.0\ (R_C)$", ax=ax_mix, color=palette[len(p_stars)+1])

        # Plot violins of bootstrapped estimates
        for p, p_star in enumerate(p_stars):
            print(f"p_C = {p_star} [#{p}/{len(p_stars)}]; ", end='', flush=True)

            # Add annotations for p_C
            ax_vio.axvline(x=p_star, ymin=0, ymax=1, ls='--', lw=3, zorder=0, color=palette[p+1])
                       #label="Ground Truth: {:3.2}".format(p_star))

            # Add shading around the true values
            if False:
                shade_span = 0.02
                ax_vio.axvspan(p_star-shade_span, p_star+shade_span, alpha=0.5, zorder=0, color=palette[p+1])


            # Select estimates at p_star and size for all methods
            df_b = df_boots[np.isclose(p_star, df_boots["p_C"]) &
                            (df_boots['Size'] == size) &
                            (df_boots["Mix"] == selected_mix)]


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
            # ax_vio.set(xlim=(0, 1))
            ax_vio.set(xlim=(-0.02, 1.02))

            if add_ci:  # Add confidence intervals
#                # The true value will be within these bars for 95% of samples (not measures)
#                # For alpha = 0.05, the CI bounds are +/- 1.96*SEM
#                df_means = df_b.groupby('Method').mean()
#                errors = np.zeros(shape=(2, len(methods)))
#                means = []
#                initials = []
#                for midx, method in enumerate(dpe._ALL_METHODS_):  # enumerate(methods):
#
#                    mean_est = df_means.loc[method, 'Estimate']
#
#                    initial = df_point[np.isclose(p_star, df_point["p_C"])
#                                       & (df_point['Size'] == size)
#                                       & (df_point["Mix"] == selected_mix)][method].values[0]
#
#                    df_piv = df_b.pivot_table(values="Estimate",
#                                            index=["p_C", "Size", "Mix", "Boot"],
#                                            columns="Method")
#
#                    ci_low, ci_upp = calc_conf_intervals(df_piv[method], initial=initial, average=np.mean, alpha=0.05, ci_method=CI_METHOD)
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

                # Extract point estimates for the particular hyperparameters
                df_p = df_point[np.isclose(p_star, df_point["p_C"])
                                & (df_point['Size'] == size)
                                & (df_point["Mix"] == selected_mix)]

                df_p_piv = df_p.drop(columns=["p_C", "Size", "Mix"])

                df_b_piv = df_b.pivot_table(values="Estimate",
                                            index=["p_C", "Size", "Mix", "Boot"],
                                            columns="Method")

                df_b_piv = df_b_piv[df_p_piv.columns]  # Manually sort before merge
                df_pe = pd.concat([df_p_piv, df_b_piv], ignore_index=True)
                errors, centres = get_error_bars(df_pe, summary=summaries[si][selected_mix][p],
                                                #  scores=scores, bins=bins, methods=None,  # TODO: prepared_methods
                                                 correct_bias=correct_bias, average=average,
                                                 alpha=alpha, ci_method=ci_method)


                if correct_bias:
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

                if correct_bias:
                    ax_vio.plot(x, y, '*', c=palette[p+1], markersize=12, markeredgecolor=(0.45, 0.45, 0.45), label="Corrected", zorder=20)
                else:
                    ax_vio.plot(x, y, 'o', c=palette[p+1], markersize=9, markeredgecolor=(0.45, 0.45, 0.45), zorder=20)  # label=r"$\hat{p}_C$", 
                    ax_vio.plot(x, y, '.', c=(0.45, 0.45, 0.45), markersize=2, markeredgecolor=(0.45, 0.45, 0.45), label=r"$\hat{p}_C$", zorder=30)

        print()
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
        sns.despine(ax=ax_vio, top=True, bottom=False, left=True, right=True, trim=True) #trim=False)
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
    # fig_select.savefig(os.path.join(fig_dir, f'violin_selection_{selected_mix}_{data_label}.png'))
    # fig_select.savefig(os.path.join(fig_dir, f'violin_selection_{selected_mix}_{data_label}.svg'), transparent=True)
    return fig_select


def plot_roc(scores, bins, title=None, full_labels=True, ax=None):
    """Plot the Reciever Operator characteristic

    Args:

    """
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    if not ax:
        fig, ax = plt.subplots()
        # plt.sca(ax)

    fpr, tpr = get_fpr_tpr(scores, bins)
    roc_auc = auc(fpr, tpr)
    # NOTE: This gives the same results as above for Coeliac and 
    # Renal applications but is slower since it uses raw scores not binned data
    # fpr, tpr, roc_auc = get_roc_scores(scores)

    # ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')  # lw=1, 
    sns.lineplot(x=fpr, y=tpr, label=f'AUC = {roc_auc:.2f}', ax=ax)
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if full_labels:
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
    else:
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    # if not title:
    #     ax.set_title('Receiver operating characteristic')
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set(aspect="equal")

    return ax
