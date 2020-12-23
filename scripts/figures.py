#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:04:57 2018

Generate manuscript figures.

@author: ben
"""

import os
import time
import warnings

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import dpe
from dpe.utilities import construct_mixture, format_seconds, load_accuracy  # get_fpr_tpr,
from dpe.datasets import (load_diabetes_data, load_renal_data, load_coeliac_data)
from dpe.plots import (plot_roc, plot_distributions, plot_bootstraps,
                       plot_characterisation, plot_selected_violins, 
                       get_error_bars)
# from dpe.config import adjust_excess, ci_method, correct_bias


# ---------------------------- Define constants ------------------------------

FRESH_DATA = True  # False  # CAUTION!
seed = 42
n_boot = 10 #1000
n_mix = 10 #100
sample_size = 1000  # -1
n_seeds = 1
selected_mix = 0
verbose = False

# Set method details
alpha = 0.05  # Alpha for confidence intervals
ci_method = "bca"  # "experimental"  # "stderr" # "centile" "jeffreys"
correct_bias = False  # Flag to use bias correction: corrected = 2 * pe_point - mean(pe_boot)
adjust_excess = False  # TODO: Reimplement this or remove?
# KDE_kernel = 'gaussian'  # ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# Configure plots
output_diabetes_rocs = False
output_application = {'Diabetes': False, 'Renal': False, 'Coeliac': True}
application_xlims = {'Diabetes': None, 'Renal': None, 'Coeliac': (0, 0.3)}
output_analysis = {'Diabetes': True, 'Renal': False, 'Coeliac': False}
output_characterisation = {'Diabetes': False, 'Renal': False, 'Coeliac': False}
# average = np.mean
# deviation = np.std

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

# Suppress warnings
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message="The 'normed' kwarg is deprecated")

# Create output directories
# out_dir = "results_test"
# out_dir = "results_100_100"
# out_dir = "results/m1000_b0"
# out_dir = "results/m1000_b10"
# out_dir = "results/m100_b1000"
# out_dir = "results/m10_b100"
# out_dir = "results/m100_b100"
# out_dir = "results/m1000_b100"
# out_dir = "results/s1000_m0_b0"
characterisation_dir = os.path.join("results", "characterisation")
if seed is None:
    seed = np.random.randint(np.iinfo(np.int32).max)
assert 0 <= seed < np.iinfo(np.int32).max
out_dir = os.path.join("results", f"n{sample_size}_m{n_mix}_b{n_boot}_s{seed}")
fig_dir = os.path.join(out_dir, "figs")
os.makedirs(fig_dir, exist_ok=True)

# ----------------------------------------------------------------------------

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

    if output_diabetes_rocs:
        # Plot ROC curves
        fig, axes = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(18, 12))
        # fig, axes = plt.subplots(3,3)
        (scores, bins, means, medians, p_C) = load_diabetes_data('T1GRS')
        # scores['R_C'] = np.append(scores['R_C'], np.random.randn(100000)*0.1+0.8)
        plot_roc(scores, bins, title='Diabetes: T1GRS', ax=axes[0, 0])
        plot_distributions(scores, bins, 'Diabetes: T1GRS', norm=True, despine=False, ax=axes[1, 0])
        (scores, bins, means, medians, p_C) = load_diabetes_data('T2GRS')
        # scores['R_C'] = np.append(scores['R_C'], np.random.randn(100000)+5)
        plot_roc(scores, bins, title='Diabetes: T2GRS', ax=axes[0, 1])
        plot_distributions(scores, bins, 'Diabetes: T2GRS', norm=True, despine=False, ax=axes[1, 1])
        (scores, bins, means, medians, p_C) = load_renal_data()
        # scores['R_C'] = np.append(scores['R_C'], np.random.randn(100000)+10)
        plot_roc(scores, bins, title='Renal', ax=axes[0, 2])
        plot_distributions(scores, bins, 'Renal', norm=True, despine=False, ax=axes[1, 2])
        fig.savefig(os.path.join(fig_dir, 'roc_Diabetes.png'))
        # exit()

    sns.set_style("ticks")

    methods = {method: True for method in dpe._ALL_METHODS_}
    if adjust_excess:
        adjustment_factor = 1/0.92  # adjusted for fact it underestimates by 8%
    else:
        adjustment_factor = 1.0


    for data_label, data in [("Diabetes", load_diabetes_data('T1GRS')),
                            #  ("Renal", load_renal_data()),
                             ("Coeliac", load_coeliac_data())]:

        (scores, bins, means, medians, p_C) = data


        if output_application[data_label]:

            res_file = os.path.join(out_dir, f"pe_results_{data_label}.pkl")

            if FRESH_DATA:  # or True:
                print(f"Running mixture analysis on {data_label} scores...", flush=True)
                t = time.time()  # Start timer

                summary, df_pe = dpe.analyse_mixture(scores, bins, methods,
                                        n_boot=n_boot, boot_size=-1, n_mix=n_mix, # boot_size=sample_size,
                                        alpha=alpha, true_pC=p_C, 
                                        ci_method=ci_method,
                                        correct_bias=correct_bias, n_jobs=-1,  # Previously correct_bias defaulted to False
                                        logfile=os.path.join(out_dir, f"pe_{data_label}.log"))

                elapsed = time.time() - t
                print(f'Elapsed time = {elapsed:.3f} seconds\n')

                # Save results
                df_pe.to_pickle(res_file)
            else:
                print(f"Loading {data_label} analysis...", flush=True)
                if os.path.isfile(res_file):
                    df_pe = pd.read_pickle(res_file)
                else:
                    warnings.warn(f"Missing data file: {res_file}")
                    break


            # Plot worked examples
            print(f"Plotting application with {data_label} scores...", flush=True)
            fig_ex = plt.figure(figsize=(12, 4))
            gs = plt.GridSpec(nrows=1, ncols=2, hspace=0.15, wspace=0.15,
                            left=0.08, right=0.95, bottom=0.15, top=0.96)

            # sns.set_style("ticks")
            with sns.axes_style("ticks"):
                ax_dists_ex = fig_ex.add_subplot(gs[0, 0])
                plot_distributions(scores, bins, data_label, ax=ax_dists_ex)

            # with sns.axes_style("whitegrid"):
            with sns.axes_style("ticks", {"axes.grid": True, "axes.spines.left": False, 'ytick.left': False}):
                ax_ci_ex = fig_ex.add_subplot(gs[0, 1])
                plot_bootstraps(df_pe, correct_bias=correct_bias, p_C=p_C, 
                                ax=ax_ci_ex, limits=application_xlims[data_label], 
                                ci_method=ci_method, initial=False, legend=False, 
                                violins=True, orient='h')
                # if application_xlims[data_label]:
                #     ax_ci_ex.set_xlim(*application_xlims[data_label])

            fig_ex.savefig(os.path.join(fig_dir, f'application_{data_label}.png'))
            fig_ex.savefig(os.path.join(fig_dir, f'application_{data_label}.svg'), transparent=True)


            
        if output_characterisation[data_label]:

            #if FRESH_DATA:
            #    exec(open("./bootstrap.py").read())

            # Load bootstraps of accurarcy data
            (point_estimates, boots_estimates, proportions, sample_sizes) = load_accuracy(characterisation_dir, data_label)

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



        if output_analysis[data_label]:

            # Plot violins for a set of proportions
            # p_stars = [0.05, 0.25, 0.50, 0.75, 0.95]
            # sizes = [100, 500, 1000, 5000, 10000]

            # p_stars = [0.25, 0.50, 0.75]
            # p_stars = [0.1, 0.50, 0.75]
            p_stars = [0.1, 0.4, 0.8]
            sizes = [500, 1000, 5000]
            # n_boot = 5

            # Generate multiple mixes
            point_estimates_res_file = os.path.join(out_dir, f"pe_stack_analysis_point_{data_label}.pkl")
            boot_estimates_res_file = os.path.join(out_dir, f"pe_stack_analysis_{data_label}.pkl")
            if FRESH_DATA:  # or True:
                print(f"Running mixture analysis with {data_label} scores...", flush=True)
                t = time.time()  # Start timer

                # Split the references distributions to ensure i.i.d. data for 
                # constructing the mixtures and estimating them.
                n_R_C, n_R_N = len(scores['R_C']), len(scores['R_N'])
                partition_R_C, partition_R_N = n_R_C//2, n_R_N//2
                inds_R_C = np.random.permutation(n_R_C)
                inds_R_N = np.random.permutation(n_R_N)
                hold_out_scores = {'R_C': scores['R_C'][inds_R_C[:partition_R_C]],
                                'R_N': scores['R_N'][inds_R_N[:partition_R_N]]}
                violin_scores = {'R_C': scores['R_C'][inds_R_C[partition_R_C:]],
                                'R_N': scores['R_N'][inds_R_N[partition_R_N:]]}

                dfs_point = []
                dfs_boot = []

                size_bar = tqdm.tqdm(sizes, dynamic_ncols=True)
                for s, size in enumerate(size_bar):
                    size_bar.set_description(f"Size = {size:6,}")
                    Mixtures = {mix: {} for mix in range(n_seeds)}

                    for mix in tqdm.trange(n_seeds, dynamic_ncols=True, desc=" Mix"):  # Redundant loop
                        mix_dist_file = os.path.join(out_dir, f"mix{mix}_size{size}_{data_label}.pkl")

                        prop_bar = tqdm.tqdm(p_stars, dynamic_ncols=True)
                        for p, p_star in enumerate(prop_bar):
                            prop_bar.set_description(f" p_C = {p_star:6.2f}")

                            violin_scores['Mix'] = construct_mixture(hold_out_scores['R_C'], hold_out_scores['R_N'], p_star, size)
                            Mixtures[mix][p_star] = violin_scores['Mix']
                            summary, df_cm = dpe.analyse_mixture(violin_scores, bins, methods,
                                                    n_boot=n_boot, boot_size=size,
                                                    n_mix=n_mix,
                                                    alpha=alpha, true_pC=p_star,
                                                    ci_method=ci_method,
                                                    correct_bias=correct_bias,  # Previously correct_bias defaulted to False
                                                    n_jobs=-1, verbose=0,
                                                    logfile=os.path.join(out_dir, f"pe_{data_label}_size_{size:05d}_p_C_{p_star:3.2f}.log"))
                            df_point = df_cm.iloc[[0]].copy()
                            df_point['Size'] = size
                            df_point["p_C"] = p_star
                            df_point['Mix'] = mix
                            # df_point = df_point.melt(var_name='Method', id_vars=["p_C", 'Size', 'Mix'], value_name='Estimate')
                            dfs_point.append(df_point)

                            df_boots = df_cm.iloc[1:, :].copy()
                            if n_mix > 0:
                                n_bootstraps_total = n_mix * n_boot
                            else:
                                n_bootstraps_total = n_boot
                            df_boots['Size'] = size * np.ones(n_bootstraps_total, dtype=int)
                            df_boots["p_C"] = p_star * np.ones(n_bootstraps_total)
                            df_boots['Mix'] = mix * np.ones(n_bootstraps_total, dtype=int)
                            df_boots["Boot"] = list(range(n_bootstraps_total))
                            df_boots = df_boots.melt(var_name='Method',
                                                    id_vars=["p_C", 'Size', 'Mix', "Boot"],
                                                    value_name='Estimate')
                            dfs_boot.append(df_boots)

                        df_size = pd.DataFrame(Mixtures[mix], columns=p_stars)
                        df_size.to_pickle(mix_dist_file)

                df_point = pd.concat(dfs_point, ignore_index=True)
                df_boots = pd.concat(dfs_boot, ignore_index=True)
                elapsed = time.time() - t
                print(f'Elapsed time = {format_seconds(elapsed)}\n')

                # Save results
                df_point.to_pickle(point_estimates_res_file)
                df_boots.to_pickle(boot_estimates_res_file)
            else:
                print(f"Loading mixture analysis with {data_label} scores...", flush=True)
                if os.path.isfile(point_estimates_res_file):
                    df_point = pd.read_pickle(point_estimates_res_file)
                else:
                    warnings.warn(f"Missing data file: {point_estimates_res_file}")
                    break
                if os.path.isfile(boot_estimates_res_file):
                    df_boots = pd.read_pickle(boot_estimates_res_file)
                else:
                    warnings.warn(f"Missing data file: {boot_estimates_res_file}")
                    break


            # Plot selected violins
            print("Plotting violins of constructed mixtures with {} scores...".format(data_label), flush=True)
            plot_mixes = [0]
            for mix in plot_mixes:  #range(n_seeds):
                fig = plot_selected_violins(scores, bins, df_point, df_boots, methods, p_stars, sizes,
                                    out_dir, data_label, selected_mix=mix,
                                    add_ci=True, alpha=0.05, ci_method=ci_method,
                                    correct_bias=correct_bias)
                fig.savefig(os.path.join(fig_dir, f'violin_selection_{mix}_{data_label}.png'))
                fig.savefig(os.path.join(fig_dir, f'violin_selection_{mix}_{data_label}.svg'), transparent=True)








        # Plot selected violins
        if False:
            #for mix in range(n_seeds):
            mix = selected_mix
            fig_vio = plt.figure(figsize=(8, 2*len(sizes)))
            ax_vio = fig_vio.add_subplot(111)

            for p, p_star in enumerate(p_stars):
                # df = df_boots[np.isclose(p_star, df_boots["p_C"]) & np.isclose(mix, df_boots['Mix'])]
                df = df_boots[np.isclose(p_star, df_boots["p_C"]) & (df_boots['Mix'] == mix)]
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
                    df = df_boots[np.isclose(p_star, df_boots["p_C"])]  # & np.isclose(size, df_vio['Size'])]
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
                    df = df_boots[(df_boots.Size == size) & np.isclose(df_boots["p_C"], p_star)]
                    
                    # Add confidence intervals

                    # df_means = df.groupby('Method').mean()
                    # errors = np.zeros(shape=(2, len(methods)))
                    # means = []
                    # for midx, method in enumerate(methods):
                    #     nobs = size  # len(df_boots[method])
                    #     mean_est = df_means.loc[method, 'Estimate']
                    #     means.append(mean_est)
                    #     # mean_est = np.mean(df_boots[method])
                    #     count = int(mean_est*nobs)
                    #     ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method=ci_method)
                    #     errors[0, midx] = mean_est - ci_low1  # y[midx] - ci_low1
                    #     errors[1, midx] = ci_upp1 - mean_est  # ci_upp1 - y[midx]

                    errors, centres = get_error_bars(df, correct_bias=correct_bias, alpha=alpha, ci_method=ci_method)  # average=average, 

                    # Add white border around error bars
                    # ax.errorbar(x=x, y=y, yerr=errors, fmt='none', c='w', lw=6, capsize=14, capthick=6)
                    if orient == 'v':
                        x = ax_vio.get_xticks()
                        y = centres  # means
                        ax_vio.errorbar(x=x, y=y, yerr=errors, fmt='s', markersize=7, c=c, lw=4, capsize=12, capthick=4, label="Confidence Intervals ({:3.1%})".format(1-alpha))
                    elif orient == 'h':
                        x = centres  # means
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
                        sns.distplot(scores["R_C"], bins=bins['edges'], hist=False, kde=True, kde_kws={"shade": True}) # hist=True, norm_hist=True, kde=False)#,
                                     # label=r"$R_1$", ax=ax_mixes)
                        for p, p_star in enumerate(p_stars):
                            sns.distplot(df_mixes[p_star], bins=bins['edges'], hist=False,
                                         label=r"$p_1^*={}$".format(p_star), ax=ax_mixes)  # \tilde{{M}}: n={},
                        sns.distplot(scores["R_N"], bins=bins['edges'], hist=False, kde=True, kde_kws={"shade": True})#,
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
        # if p_C is not None:
        #     errors[]
        # if p_C is None:
        #     p_C = np.mean()

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

                plot_bootstraps(df_pe, correct_bias, p_C, axes[b], limits=(0, 1), ci_method=ci_method, alpha=alpha, legend=legend, orient='h')

            fig.savefig(os.path.join(fig_dir, "boot_size_{}.png".format(data_label)))

            for mix in range(n_seeds):
                df_mix = df_boots[(df_boots['Mix'] == mix)]
                frames = []

                for b in [1000, 100, 10, 1]:
                    for s, size in enumerate(sizes):
                        for p, p_star in enumerate(p_stars):
                            df_ps = df_mix[np.isclose(p_star, df_mix["p_C"])
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
                    g = sns.catplot(x="Bootstraps", y="Error", hue="Method", col="p_C", row="Size", row_order=sizes[::-1], data=df, kind="violin") #, margin_titles=True) #, ax=ax_err)
                    g.set(ylim=(-0.5, 0.5)).despine(left="True")
                    # g.despine(left="True")

                plt.savefig(os.path.join(fig_dir, "bootstraps_{}_{}.png".format(mix, data_label)))
