#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

Module to analyse an unknown mixture population.

@author: ben
"""

from pprint import pprint
# import warnings

import numpy as np
import pandas as pd
import lmfit
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange
from statsmodels.stats.proportion import proportion_confint
from sklearn.neighbors import KernelDensity
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=DeprecationWarning)
#     from sklearn.neighbors import KernelDensity
# from statsmodels.nonparametric.kde import KDEUnivariate
# TODO: replace with a scipy/numpy function to reduce dependencies
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

# TODO: Try replacing with scipy.optimize.curve_fit to reduce dependencies:
# https://lmfit.github.io/lmfit-py/model.html


METHODS_ORDER = ["Excess", "Means", "EMD", "KDE"]


def fit_kernels(scores, bw):
    kernels = {}
    for label, data in scores.items():
        kernels[label] = {}
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:

            kernels[label][kernel] = KernelDensity(kernel=kernel, bandwidth=bw,
                                                   atol=0, rtol=1e-4).fit(X)
    return kernels


# @mem.cache
def fit_KDE_model(Mix, bins, model, params_mix, kernel):
    # model = methods["model"]
    x_KDE = bins["centers"]
    kde_mix = KernelDensity(kernel=kernel, bandwidth=bins['width'])
    kde_mix.fit(Mix[:, np.newaxis])
    res_mix = model.fit(np.exp(kde_mix.score_samples(x_KDE[:, np.newaxis])),
                        x=x_KDE, params=params_mix)
    amp_Ref1 = res_mix.params['amp_1'].value
    amp_Ref2 = res_mix.params['amp_2'].value
    return amp_Ref1 / (amp_Ref1 + amp_Ref2)


def interpolate_CDF(scores, x_i, min_edge, max_edge):
    # TODO: x = [x_i[0], *sorted(scores), x_i[-1]]
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def prepare_methods_(methods, scores, bins, verbose=1):

    if "Excess" in methods:
        if not isinstance(methods["Excess"], dict):
            methods["Excess"] = {}

        # TODO: CHECK!!! This should be the "healthy" reference population
        methods["Excess"].setdefault("median", np.median(scores["Ref2"]))
        methods["Excess"].setdefault("adj_factor", 1)

    if "Means" in methods:
        if not isinstance(methods["Means"], dict):
            methods["Means"] = {"mu_1": np.mean(scores["Ref1"]),
                                "mu_2": np.mean(scores["Ref2"])}

    if "EMD" in methods:
        if not isinstance(methods["EMD"], dict):
            methods["EMD"] = {}
            methods["EMD"]["max_EMD"] = bins["max"] - bins["min"]

            # Interpolate the cdfs at the same points for comparison
            CDF_1 = interpolate_CDF(scores["Ref1"], bins['centers'],
                                    bins['min'], bins['max'])
            methods["EMD"]["CDF_1"] = CDF_1

            CDF_2 = interpolate_CDF(scores["Ref2"], bins['centers'],
                                    bins['min'], bins['max'])
            methods["EMD"]["CDF_2"] = CDF_2

            # EMDs computed with interpolated CDFs
            methods["EMD"]["EMD_1_2"] = sum(abs(CDF_1 - CDF_2))

    if "KDE" in methods:
        if not isinstance(methods["KDE"], dict):
            methods["KDE"] = {}
        methods["KDE"].setdefault("kernel", "gaussian")
        methods["KDE"].setdefault("bandwidth", bins["width"])

        if "model" not in methods["KDE"]:
            kdes = fit_kernels(scores, methods["KDE"]["bandwidth"])
            kde_1 = kdes["Ref1"][methods["KDE"]["kernel"]]
            kde_2 = kdes["Ref2"][methods["KDE"]["kernel"]]

            # Assigning a default value to amp initialises them
            # x := Bin centres
            def dist_1(x, amp_1=1):
                return amp_1 * np.exp(kde_1.score_samples(x[:, np.newaxis]))

            def dist_2(x, amp_2=1):
                return amp_2 * np.exp(kde_2.score_samples(x[:, np.newaxis]))

            methods["KDE"]["model"] = lmfit.Model(dist_1) + lmfit.Model(dist_2)

        if "params" not in methods["KDE"]:
            methods["KDE"]["params"] = methods["KDE"]["model"].make_params()
            methods["KDE"]["params"]["amp_1"].value = 1
            methods["KDE"]["params"]["amp_1"].min = 0
            methods["KDE"]["params"]["amp_2"].value = 1
            methods["KDE"]["params"]["amp_2"].min = 0


def prepare_methods(methods, scores, bins, verbose=1):

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']
    bin_width = bins['width']
    bin_edges = bins['edges']

    # TODO: Consolidate kwargs into methods
    kwargs = {}
    # kwargs['bins'] = bins

    # ----------------------------- Excess method -----------------------------
    if "Excess" in methods:

        if isinstance(methods["Excess"], dict):
            if "Median_Ref1" not in methods["Excess"]:
                methods["Excess"]["Median_Ref1"] = np.median(scores["Ref1"])
            if "Median_Ref2" not in methods["Excess"]:
                methods["Excess"]["Median_Ref2"] = np.median(scores["Ref2"])
            median = methods["Excess"]["Median_Ref2"]
            if "adjustment_factor" not in methods["Excess"]:
                methods["Excess"]["adjustment_factor"] = 1
            kwargs['adjustment_factor'] = methods["Excess"]["adjustment_factor"]
    #        if isinstance(methods["Excess"], float):
    #            # Median passed
    #            median = methods["Excess"]
    #            print("Passed median: {}".format(median))
    #        else:
    #            # The Excess method assumes that...
    #            median = np.median(scores["Ref2"])

        # TODO: Check this!!!
        kwargs['population_median'] = median

        if verbose > 1:
            print("Ref1 median:", np.median(Ref1))
            print("Ref2 median:", np.median(Ref2))
            print("Population median: {}".format(median))
            print("Mixture size:", len(Mix))  # boot_size)

    # ----------------------------- Means method ------------------------------
    if "Means" in methods:
        try:
            Mean_Ref1 = methods["Means"]["Ref1"]
        except (KeyError, TypeError):
            if verbose > 1:
                print("No Mean_Ref1 specified!")
            Mean_Ref1 = Ref1.mean()
        finally:
            kwargs["Mean_Ref1"] = Mean_Ref1
        try:
            Mean_Ref2 = methods["Means"]["Ref2"]
        except (KeyError, TypeError):
            if verbose > 1:
                print("No Mean_Ref2 specified!")
            Mean_Ref2 = Ref2.mean()
        finally:
            kwargs["Mean_Ref2"] = Mean_Ref2

    # ------------------------------ EMD method -------------------------------
    if "EMD" in methods:

        if 'max_EMD' not in kwargs:
            max_EMD = bin_edges[-1] - bin_edges[0]
            kwargs['max_EMD'] = max_EMD

        # Interpolate the cdfs at the same points for comparison
        if 'i_CDF_Ref1' not in kwargs:
            i_CDF_Ref1 = interpolate_CDF(Ref1, bins['centers'],
                                         bins['min'], bins['max'])
            kwargs['i_CDF_Ref1'] = i_CDF_Ref1

        if 'i_CDF_Ref2' not in kwargs:
            i_CDF_Ref2 = interpolate_CDF(Ref2, bins['centers'],
                                         bins['min'], bins['max'])
            kwargs['i_CDF_Ref2'] = i_CDF_Ref2
    #        i_CDF_Mix = interpolate_CDF(Mix, bins['centers'], bins['min'], bins['max'])

        # EMDs computed with interpolated CDFs
        if 'i_EMD_1_2' not in kwargs:
            i_EMD_1_2 = sum(abs(kwargs['i_CDF_Ref1']-kwargs['i_CDF_Ref2']))
            kwargs['i_EMD_1_2'] = i_EMD_1_2
    #        i_EMD_21 = sum(abs(i_CDF_Ref2-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M1 = sum(abs(i_CDF_Mix-i_CDF_Ref1)) * bin_width / max_EMD
    #        i_EMD_M2 = sum(abs(i_CDF_Mix-i_CDF_Ref2)) * bin_width / max_EMD
    #        kwargs['i_EMD_21'] = i_EMD_21

    # ------------------------------ KDE method ------------------------------
    if "KDE" in methods:

        bw = bin_width  # Bandwidth

        if 'kdes' not in kwargs:
            kdes = fit_kernels(scores, bw)
            kwargs['kdes'] = kdes
        else:
            kdes = kwargs['kdes']

        if 'KDE_kernel' not in kwargs or 'bin_width' not in kwargs:
            try:
                KDE_kernel = methods["KDE"]["kernel"]
            except (KeyError, TypeError):
                if verbose > 1:
                    print("No kernel specified!")
                KDE_kernel = "gaussian"  # Default kernel
            else:
                try:
                    bw = methods["KDE"]["bandwidth"]
                except (KeyError, TypeError):
                    bw = bins["width"]
            finally:
                if verbose > 1:
                    print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))
            kwargs['KDE_kernel'] = KDE_kernel
            kwargs['bin_width'] = bin_width
        else:
            KDE_kernel = kwargs['KDE_kernel']

        if 'model' not in kwargs:
            # Define the KDE models
            # x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
            # Assigning a default value to amp initialises them
            def kde_Ref1(x, amp_Ref1=1):
                return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

            def kde_Ref2(x, amp_Ref2=1):
                return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))

            kwargs['model'] = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

        if 'initial_params' not in kwargs:
            params_mix = kwargs['model'].make_params()
            params_mix['amp_Ref1'].value = 1
            params_mix['amp_Ref1'].min = 0
            params_mix['amp_Ref2'].value = 1
            params_mix['amp_Ref2'].min = 0
            kwargs['initial_params'] = params_mix

    return kwargs


def generate_report(df_pe, true_p1=None, alpha=0.05):
    n_boot = len(df_pe)
    report = []
    report.append("{:20} | {:^17s} | {:^17s} ".format("Proportion Estimates",
                                                      "Reference 1",
                                                      "Reference 2"))
    report.append("="*61)
    for method in df_pe:  # loop over columns (i.e. methods)
        values = df_pe[method]
#        print("{:20} | {:<17.5f} | {:<17.5f} ".format(method, initial_results[method], 1-initial_results[method]))
        report.append(" {:13} (µ±σ) | {:.5f} +/- {:.3f} | {:.5f} +/- {:.3f} "
                      .format(method, np.mean(values), np.std(values),
                              1-np.mean(values), np.std(1-values)))  # (+/- SD)
        if n_boot > 1:
            nobs = len(values)
            count = int(np.mean(values)*nobs)
            # TODO: Multiply the numbers by 10 for more accuracy 45/100 --> 457/1000 ?
            ci_low1, ci_upp1 = proportion_confint(count, nobs, alpha=alpha, method='normal')
            ci_low2, ci_upp2 = proportion_confint(nobs-count, nobs, alpha=alpha, method='normal')
            report.append(" C.I. (level={:3.1%})  | {:.5f},  {:.5f} | {:.5f},  {:.5f} "
                          .format(1-alpha, ci_low1, ci_upp1, ci_low2, ci_upp2))
        report.append("-"*61)
    if true_p1:
        report.append(" {:19} | {:<17.5f} | {:<17.5f} "
                      .format("Ground Truth", true_p1, 1-true_p1))
        report.append("="*61)
    # report.append("\n")
    return "\n".join(report)


def analyse_mixture(scores, bins, methods, n_boot=1000, boot_size=-1,
                    alpha=0.05, true_p1=None, n_jobs=1, seed=None,
                    verbose=1, logfile='', kwargs=None):

    if seed is not None:
        np.random.seed(seed)

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']

    if kwargs is None:
        kwargs = prepare_methods_(methods, scores, bins, verbose=verbose)

    def estimate_Ref1_(RM, Ref1, Ref2, bins, methods=None):
        '''Estimate the proportion of two reference populations comprising
        an unknown mixture.

        The returned proportions are with respect to Ref_1.
        The proportion of Ref_2, p_2, is assumed to be 1 - p_1.
        '''

        if methods is None:
            methods = {method: True for method in METHODS_ORDER}
        # bins = kwargs['bins']
        results = {}

        # ------------------------- Subtraction method ------------------------
        if "Excess" in methods:
            # Calculate the proportion of another population w.r.t. the excess
            # number of cases from the mixture's assumed majority population.
            # Ref1: disease; Ref2: healthy

            number_low = len(RM[RM <= methods["Excess"]["median"]])
            number_high = len(RM[RM > methods["Excess"]["median"]])
            p1_est = abs(number_high - number_low) / len(RM)

            p1_est *= methods["Excess"]["adj_factor"]
            results['Excess'] = np.clip(p1_est, 0.0, 1.0)

        # --------------------- Difference of Means method --------------------
        if "Means" in methods:

            mu_1, mu_2 = methods["Means"]["mu_1"], methods["Means"]["mu_2"]
            if mu_1 > mu_2:
                p1_est = (RM.mean() - mu_2) / (mu_1 - mu_2)
            else:
                p1_est = (mu_2 - RM.mean()) / (mu_2 - mu_1)

            # TODO: Check!
            # p1_est = abs((RM.mean() - mu2) / (mu1 - mu2))
            results['Means'] = np.clip(p1_est, 0.0, 1.0)

        # ----------------------------- EMD method ----------------------------
        if "EMD" in methods:
            # Interpolated cdf (to compute EMD)
            CDF_Mix = interpolate_CDF(RM, bins['centers'], bins['min'], bins['max'])
            EMD_M_1 = sum(abs(CDF_Mix - methods["EMD"]["CDF_1"]))
            EMD_M_2 = sum(abs(CDF_Mix - methods["EMD"]["CDF_2"]))
            results["EMD"] = 0.5 * (1 + (EMD_M_2 - EMD_M_1) / methods["EMD"]["EMD_1_2"])

        # ----------------------------- KDE method ----------------------------
        if "KDE" in methods:
            # TODO: Print out warnings if goodness of fit is poor?
            results['KDE'] = fit_KDE_model(RM, bins, methods["KDE"]['model'],
                                           methods["KDE"]["params"],
                                           methods["KDE"]["kernel"])

        return results

    def estimate_Ref1(RM, Ref1, Ref2, bins, methods, kwargs):
        '''Estimate the proportion of two reference populations comprising
        an unknown mixture.

        The returned proportions are with respect to Ref_1.
        The proportion of Ref_2, p_2, is assumed to be 1 - p_1.
        '''

        # bins = kwargs['bins']
        results = {}

        # ------------------------- Subtraction method ------------------------
        if "Excess" in methods:
            # Calculate the proportion of another population w.r.t. the excess
            # number of cases from the mixture's assumed majority population.

            # Ref1: disease; Ref2: healthy
            # if medians["Ref1"] > medians["Ref2"]:
            #    excess_cases = RM[RM > kwargs['population_median']].count()
            #    expected_cases = RM[RM <= kwargs['population_median']].count()
            # else:  # disease has a lower median than the healthy population
            #    excess_cases = RM[RM <= kwargs['population_median']].count()
            #    expected_cases = RM[RM > kwargs['population_median']].count()
            # boot_size = len(RM)
            # results['Excess'] = abs(excess_cases - expected_cases)/boot_size

            number_low = len(RM[RM <= kwargs['population_median']])
            number_high = len(RM[RM > kwargs['population_median']])
            boot_size = len(RM)

            results['Excess'] = abs(number_high - number_low)/boot_size
            results['Excess'] *= methods["Excess"]["adjustment_factor"]

            # Clip results
            results['Excess'] = np.clip(results['Excess'], 0.0, 1.0)

        # --------------------- Difference of Means method --------------------
        if "Means" in methods:

            mu1, mu2 = kwargs['Mean_Ref1'], kwargs['Mean_Ref2']
            if mu1 > mu2:
                p1_est = (RM.mean() - mu2) / (mu1 - mu2)
            else:
                p1_est = (mu2 - RM.mean()) / (mu2 - mu1)

            # TODO: Check!
            # p1_est = abs((RM.mean() - mu2) / (mu1 - mu2))
            results['Means'] = np.clip(p1_est, 0.0, 1.0)

        # ----------------------------- EMD method ----------------------------
        if "EMD" in methods:
            # Interpolated cdf (to compute EMD)
            i_CDF_Mix = interpolate_CDF(RM, bins['centers'],
                                        bins['min'], bins['max'])

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

        # ----------------------------- KDE method ----------------------------
        if "KDE" in methods:
            # TODO: Print out warnings if goodness of fit is poor?
            results['KDE'] = fit_KDE_model(RM, bins, kwargs['model'],
                                           kwargs['initial_params'],
                                           kwargs['KDE_kernel'])

        return results

    def bootstrap_mixture(Mix, Ref1, Ref2, bins, methods, boot_size=-1, seed=None):  #, kwargs=None):

        if boot_size == -1:
            boot_size = len(Mix)

        if seed is None:
            bs = np.random.choice(Mix, boot_size, replace=True)
        else:
            bs = np.random.RandomState(seed).choice(Mix, boot_size, replace=True)

        return estimate_Ref1_(bs, Ref1, Ref2, bins, methods)  #, kwargs)

    columns = [method for method in METHODS_ORDER if method in methods]

    if n_boot <= 0:
        # Get initial estimate of proportions
        initial_results = estimate_Ref1_(Mix, Ref1, Ref2, bins, methods)  #, kwargs)
        if verbose > 1:
            pprint(initial_results)
        if true_p1:
            if verbose > 1:
                print("Ground truth: {:.5f}".format(true_p1))

        df_pe = pd.DataFrame(initial_results, index=[0], columns=columns)

    else:  # if n_boot > 0:
        if verbose > 0:
            if n_jobs == -1:
                nprocs = cpu_count()
            else:
                nprocs = n_jobs
            print('Running {} bootstraps with {} processors...'
                  .format(n_boot, nprocs), flush=True)
            disable = False
        else:
            disable = True
        if verbose == -1:  # Allow only progress bar
            disable = False

        # Make bootstrapping deterministic with parallelism
        # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
        boot_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_boot)

        # HACK: This is a kludge to reduce the joblib overhead when n_jobs=1
        if n_jobs == 1 or n_jobs is None:
            # NOTE: These results are identical to when n_jobs=1 in the parallel section however it takes about 25% less time per iteration
            results = [bootstrap_mixture(Mix, Ref1, Ref2, bins, methods, boot_size, seed=None)  #, kwargs=kwargs)
                       for b in trange(n_boot, desc="Bootstraps", dynamic_ncols=True, disable=disable)]
        else:
            with Parallel(n_jobs=n_jobs) as parallel:
                results = parallel(delayed(bootstrap_mixture)(Mix, Ref1, Ref2, bins, methods, boot_size, seed=b_seed)  #, kwargs=kwargs)
                                   for b_seed in tqdm(boot_seeds, desc="Bootstraps", dynamic_ncols=True, disable=disable))

        # Put into dataframe
        df_pe = pd.DataFrame.from_records(results, columns=columns)

    # ----------- Summarise proportions for the whole distribution ------------
    report = generate_report(df_pe, true_p1=true_p1, alpha=alpha)
    if verbose > 0:
        print("\n", report, "\n")

    if logfile is not None:
        if logfile == '':
            logfile = "proportion_estimates.log"
        with open(logfile, 'w') as lf:
            lf.write(report)
            lf.write("\n")

    return df_pe
