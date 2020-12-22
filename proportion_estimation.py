#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

Module to analyse an unknown mixture population.

@author: ben
"""

# TODO: Move point_estimate and bootstrap_mixture to top level and hide other functions

from pprint import pprint
import itertools
import copy
import warnings

import numpy as np
import scipy as sp
from scipy.stats import norm
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


_ALL_METHODS_ = ["Excess", "Means", "EMD", "KDE"]


# Let's use FD!
def estimate_bins(data, bin_range=None, verbose=0):
    """Generate GRS bins through data-driven methods in `np.histogram`.
    
    These methods include:
    ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']

    For more information see: 
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    """
    # TODO: Refactor to pass one method and return only that dictionary

    # 'scott': n**(-1./(d+4))
    # kdeplot also uses 'silverman' as used by scipy.stats.gaussian_kde
    # (n * (d + 2) / 4.)**(-1. / (d + 4))
    # with n the number of data points and d the number of dimensions
    line_width = 49

    hist = {}
    bins = {}
    if verbose:
        print("  Method | Data |  n  |  width  |      range     ", flush=True)
        print("="*line_width)
    for method in ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']:
        all_scores = []
        all_refs = []
        for group, scores in data.items():
            all_scores.extend(scores)
            if group != "Mix":
                all_refs.extend(scores)
            # else:  # Add extremes to ensure the full range is spanned
            #     all_refs.extend([min(scores), max(scores)])
            if bin_range is None:
                bin_range = (min(all_scores), max(all_scores))
            if verbose > 1:
                _, bin_edges = np.histogram(scores, bins=method, range=bin_range)
                print(f" {method:>7} | {group:>4} | {len(bin_edges)-1:>3} | {bin_edges[1]-bin_edges[0]:<7.5f} | [{bin_edges[0]:5.3}, {bin_edges[-1]:5.3}]")
                # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format(group, method, bin_edges[1]-bin_edges[0], len(bin_edges)-1, bin_edges[0], bin_edges[-1]))

        h_r, edges_r = np.histogram(all_refs, bins=method,
                                    range=(min(all_scores), max(all_scores)))
        if verbose > 1:
            print("-"*line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "Refs", len(edges_r)-1, edges_r[1]-edges_r[0], edges_r[0], edges_r[-1]))

        h_a, edges_a = np.histogram(all_scores, bins=method, range=bin_range)  # Return edges

        if verbose:
            if verbose > 1:
                print("-"*line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "All", len(edges_a)-1, edges_a[1]-edges_a[0], edges_a[0], edges_a[-1]))
            # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format("All", method, b['width'], b['n'], b['min'], b['max']))
            if verbose > 1:
                print("="*line_width)
            else:
                print("-"*line_width)

        # h, edges = h_a, edges_a
        h, edges = h_r, edges_r
        hist[method] = h
        bins[method] = {'width': edges[1] - edges[0],
                        'min': edges[0],
                        'max': edges[-1],
                        'edges': edges,
                        'centers': (edges[:-1] + edges[1:]) / 2,
                        'n': len(edges) - 1}
    return hist, bins


def fit_kernel(scores, bw, kernel='gaussian'):  # , atol=0, rtol=1-4):
    """Fit kernel densities to the data."""
    X = scores[:, np.newaxis]
    return KernelDensity(kernel=kernel, bandwidth=bw, atol=0, rtol=1e-4).fit(X)


# @mem.cache
def fit_KDE_model(Mix, bins, model, params_mix, kernel, method='leastsq'):
    """Fit a combination of two reference population kernel density estimates
    to a mixture. 
    
    The amplitude of each reference population is adjust iteratively using the
    Levenberg-Marquardt (least squares) algorithm by default, to optimise
    the fit to the mixture population. The amplitudes are then normalised to 
    give the proportion of R_C (cases) within the mixture.
    """

    # model = methods["model"]
    x_KDE = bins["centers"]
    # kde_mix = KernelDensity(kernel=kernel, bandwidth=bins['width'])
    # kde_mix.fit(Mix[:, np.newaxis])
    kde_mix = fit_kernel(Mix, bw=bins['width'], kernel=kernel)
    res_mix = model.fit(np.exp(kde_mix.score_samples(x_KDE[:, np.newaxis])),
                        x=x_KDE, params=params_mix, method=method)
    amp_R_C = res_mix.params['amp_1'].value
    amp_R_N = res_mix.params['amp_2'].value
    return amp_R_C / (amp_R_C + amp_R_N)


def interpolate_CDF(scores, x_i, min_edge, max_edge):
    """Interpolate the cumulative density function of the scores at the points
    in the array `x_i`.
    """

    # TODO: x = [x_i[0], *sorted(scores), x_i[-1]]
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def prepare_methods(scores, bins, methods=None, verbose=1):
    """Extract properties of the score distributions and cache intermediate
    results for efficiency.
    """

    methods_ = copy.deepcopy(methods)  # Prevent issue with repeated runs
    if isinstance(methods_, str):
        method_name = methods_
        if method_name.lower() == 'all':
            methods_ = {method: True for method in _ALL_METHODS_}
        elif method_name.capitalize() in _ALL_METHODS_:
            methods_ = {method_name.capitalize(): True}
        else:
            warnings.warn(f'Unknown method passed: {methods_}. Running all methods...')
            methods_ = {method: True for method in _ALL_METHODS_}
    elif isinstance(methods_, (list, set)):
        methods_ = {method: True for method in methods_}
    elif isinstance(methods_, dict):
        assert "Excess" in methods_ or "Means" in methods_ or "EMD" in methods_ or "KDE" in methods_
    elif methods_ is None:  # Run all methods
        methods_ = {method: True for method in _ALL_METHODS_}
    else:
        warnings.warn(f'Unknown method passed: {methods_} ({type(methods_)}). Running all methods...')
        methods_ = {method: True for method in _ALL_METHODS_}

    if "Excess" in methods_:
        if not isinstance(methods_["Excess"], dict):
            methods_["Excess"] = {}

        # This should be the "healthy" non-cases reference population
        methods_["Excess"].setdefault("median", np.median(scores["R_N"]))
        methods_["Excess"].setdefault("adj_factor", 1)

    if "Means" in methods_:
        if not isinstance(methods_["Means"], dict):
            methods_["Means"] = {"mu_C": np.mean(scores["R_C"]),
                                 "mu_N": np.mean(scores["R_N"])}
        
        mu_C, mu_N = methods_["Means"]["mu_C"], methods_["Means"]["mu_N"]
        mix_mean = scores["Mix"].mean()
        if mix_mean < min(mu_C, mu_N) or mix_mean > max(mu_C, mu_N):
            warnings.warn(f"The mixture mean ({mix_mean:.3}) lies outside of "
                          "the range of reference means [{mu_C:.3}, {mu_N:.3}]"
                          " so is unsuitable for this mixture analysis.")

    if "EMD" in methods_:
        if not isinstance(methods_["EMD"], dict):
            methods_["EMD"] = {}
            methods_["EMD"]["max_EMD"] = bins["max"] - bins["min"]

            # Interpolate the cdfs at the same points for comparison
            CDF_1 = interpolate_CDF(scores["R_C"], bins['centers'],
                                    bins['min'], bins['max'])
            methods_["EMD"]["CDF_1"] = CDF_1

            CDF_2 = interpolate_CDF(scores["R_N"], bins['centers'],
                                    bins['min'], bins['max'])
            methods_["EMD"]["CDF_2"] = CDF_2

            # EMDs computed with interpolated CDFs
            methods_["EMD"]["EMD_1_2"] = sum(abs(CDF_1 - CDF_2))

    if "KDE" in methods_:
        if not isinstance(methods_["KDE"], dict):
            methods_["KDE"] = {}
        methods_["KDE"].setdefault("kernel", "gaussian")
        methods_["KDE"].setdefault("bandwidth", bins["width"])
        # methods_["KDE"].setdefault("atol", 0)
        # methods_["KDE"].setdefault("rtol", 1e-4)

        if "model" not in methods_["KDE"]:
            # kdes = fit_kernels(scores, methods["KDE"]["bandwidth"], methods["KDE"]["kernel"])
            # kde_1 = kdes["R_C"]  # [methods["KDE"]["kernel"]]
            # kde_2 = kdes["R_N"]  # [methods["KDE"]["kernel"]]
            kde_1 = fit_kernel(scores["R_C"], methods_["KDE"]["bandwidth"],
                               methods_["KDE"]["kernel"])
            kde_2 = fit_kernel(scores["R_N"], methods_["KDE"]["bandwidth"],
                               methods_["KDE"]["kernel"])

            # Assigning a default value to amp initialises them
            # x := Bin centres
            def dist_1(x, amp_1=1):
                return amp_1 * np.exp(kde_1.score_samples(x[:, np.newaxis]))

            def dist_2(x, amp_2=1):
                return amp_2 * np.exp(kde_2.score_samples(x[:, np.newaxis]))

            # The model assumes a linear combination of the two reference distributions only
            methods_["KDE"]["model"] = lmfit.Model(dist_1) + lmfit.Model(dist_2)

        if "params" not in methods_["KDE"]:
            methods_["KDE"]["params"] = methods_["KDE"]["model"].make_params()
            methods_["KDE"]["params"]["amp_1"].value = 1
            methods_["KDE"]["params"]["amp_1"].min = 0
            methods_["KDE"]["params"]["amp_2"].value = 1
            methods_["KDE"]["params"]["amp_2"].min = 0

    return methods_


def correct_estimate(df_pe):
    """Apply bootstrap based bias correction.
    
    Assuming that the distribution of error between the the initial point 
    estimate and the real proportion is well approximated by the distribution
    of the error between the bootstrap estimates and the initial point
    estimate.
    """

    assert len(df_pe) > 1  # Need at least one boot strap estimate
    pe_point = df_pe.iloc[0, :]
    # if len(df_pe) == 1:  # No bootstraps
    #     return pe_point
    pe_boot = df_pe.iloc[1:, :]
    n_boot = len(pe_boot)
    corrected = {}
    for method in df_pe:  # loop over columns (i.e. methods)
        # point_est = pe_point[method]
        if n_boot > 0:
            # bias = point_est - np.mean(pe_boot[method])
            # corrected[method] = point_est + bias
            corrected[method] = 2 * pe_point[method] - np.mean(pe_boot[method])
    return pd.DataFrame(corrected, index=[-1], columns=df_pe.columns)


def calc_conf_intervals(values, initial=None, correct_bias=True, average=np.mean,
                        alpha=0.05, ci_method="bca"):
    """Calculate confidence intervals for a point estimate.
    
    By default we use the alpha quantile of the distribution of the N_M * N_B
    bootstrapped p_C values, where alpha = 0.05.

    Parameters
    ----------
    values : array
        The array of bootstrapped p_C estimates (for a particular method).
    initial : float, optional
        An optional estimate from the original mixture (for a particular method). 
    correct_bias : None or bool, optional
        A flag to correct the `experimental` CI method. 
    average : function, optional
        The function used to calculate the average estimate across bootstraps. 
    alpha : float, optional
        The percentile to use for the confidence intervals (default = 0.05). 
        The returned values are (alpha/2, 1-alpha/2) percentile confidence 
        intervals.
    ci_method : str, optional
        The name of the method used to calculate the confidence intervals. 
    """

    n_obs = len(values)
    average_value = average(values)

    if ci_method.lower() == 'bca':
        # Adapted from https://github.com/cgevans/scikits-bootstrap
        # TODO: Replace with this external library for more robust checks
        
        # print("Using BCa method...")
        # Esitamate the bias correction value (the median bias transformed into normal deviates)
        z0 = norm.ppf(np.sum(values < average_value, axis=0) / n_obs)

        # Statistics of the jackknife distribution
        indices = np.arange(n_obs)
        jack_values = [average(values[indices != ind]) for ind in range(n_obs)]
        jack_mean = np.mean(jack_values)

        # Temporarily kill numpy warnings:
        oldnperr = np.seterr(invalid='ignore')
        # Acceleration value
        a = np.sum((jack_mean - jack_values)**3) / (6 * np.sum((jack_mean - jack_values)**2)**1.5)
        if np.any(np.isnan(a)):
            nanind = np.nonzero(np.isnan(a))
            warnings.warn(f"BCa acceleration values for indexes {nanind} were \
                            undefined. Statistic values were likely all equal. \
                            Affected CI will be inaccurate.")
        
        alphas = np.array([alpha/2, 1-alpha/2])

        zs = z0 + norm.ppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)
        avals = norm.cdf(z0 + zs/(1-a*zs))
        np.seterr(**oldnperr)

        values = values.to_numpy()
        values.sort(axis=0)
        nvals = np.round((n_obs-1)*avals)
        nvals = np.nan_to_num(nvals).astype('int')
        ci_low, ci_upp = values[nvals]

    elif ci_method.lower() == 'experimental':
        assert initial is not None and 0.0 <= initial <= 1.0
        if correct_bias:
            err_low, err_upp = np.percentile(values-initial, [100*alpha/2, 100*(1-alpha/2)])
            ci_low, ci_upp = initial-err_upp, initial-err_low
        else:
            # TODO: Refactor
            ci_low, ci_upp = np.percentile(values, [100*alpha/2, 100*(1-alpha/2)])
    elif ci_method.lower() == 'centile':
        ci_low, ci_upp = np.percentile(values, [100*alpha/2, 100*(1-alpha/2)])
    elif ci_method.lower() == 'stderr':
        p = average_value
        # NOTE: This currently allows CIs outside [0, 1]
        err = np.sqrt(p*(1-p)/n_obs) * sp.stats.norm.ppf(1-alpha/2)
        ci_low, ci_upp = p - err, p + err
    else:  # Assumes a binomial distribution
        count = int(average_value*n_obs)
        ci_low, ci_upp = proportion_confint(count, n_obs, alpha=alpha,
                                            method=ci_method)

    return ci_low, ci_upp


def generate_report(df_pe, true_pC=None, alpha=0.05, ci_method="bca"):
    """Generate an proportion estimate report for each method."""
    # TODO: Incorporate ipoint estimates in report
    pe_point = df_pe.iloc[0, :]
    pe_boot = df_pe.iloc[1:, :]
    n_boot = len(pe_boot)  # len(df_pe)
    line_width = 54
    report = []
    report.append(" {:^12} | {:^17s} | {:^17s} ".format("Method",
                                                        "Estimated p_C",   # Reference 1
                                                        "Estimated p_N"))  # Reference 2
    report.append("="*line_width)
    for method in df_pe:  # loop over columns (i.e. methods)
        values = pe_boot[method]
        point_est = pe_point[method]
#        print("{:20} | {:<17.5f} | {:<17.5f} ".format(method, initial_results[method], 1-initial_results[method]))
        report.append(" {:6} point | {:<17.5f} | {:<17.5f} ".format(method, pe_point[method], 1-pe_point[method]))
        report.append(" {:6} (µ±σ) | {:.5f} +/- {:.3f} | {:.5f} +/- {:.3f} "
                      .format(method, np.mean(values), np.std(values),
                              1-np.mean(values), np.std(1-values)))  # (+/- SD)

        if n_boot > 1:
            ci_low1, ci_upp1 = calc_conf_intervals(values, initial=point_est,
                                                   average=np.mean, alpha=alpha,
                                                   ci_method=ci_method)
            ci_low2, ci_upp2 = 1-ci_upp1, 1-ci_low1

            report.append(" C.I. ({:3.1%}) | {:<8.5f},{:>8.5f} | {:<8.5f},{:>8.5f} "
                          .format(1-alpha, ci_low1, ci_upp1, ci_low2, ci_upp2))
            if ci_method.lower() == "experimental":
                bias = point_est - np.mean(values)
                corrected_est = point_est + bias
                report.append(" Corrected    | {:<17.5f} | {:<17.5f} "
                              .format(corrected_est, 1-corrected_est))
        report.append("-"*line_width)
    if true_pC:
        report.append(" {:12} | {:<17.5f} | {:<17.5f} "
                      .format("Ground Truth", true_pC, 1-true_pC))
        report.append("="*line_width)
    # report.append("\n")
    return "\n".join(report)


def point_estimate(RM, R_C, R_N, bins, methods=None):
    """Estimate the proportion of two reference populations comprising
    an unknown mixture.

    The returned proportions are with respect to R_C, the cases.
    The proportion of R_N, p_N, is assumed to be 1 - p_C.
    """

    # bins = kwargs['bins']
    results = {}

    # ------------------------- Subtraction method ------------------------
    if "Excess" in methods:
        # Calculate the proportion of another population w.r.t. the excess
        # number of cases from the mixture's assumed majority population.
        # R_C: cases (disease); R_N: non-cases (healthy)

        number_low = len(RM[RM <= methods["Excess"]["median"]])
        number_high = len(RM[RM > methods["Excess"]["median"]])
        p1_est = abs(number_high - number_low) / len(RM)

        p1_est *= methods["Excess"]["adj_factor"]
        results['Excess'] = np.clip(p1_est, 0.0, 1.0)

    # --------------------- Difference of Means method --------------------
    if "Means" in methods:

        mu_C, mu_N = methods["Means"]["mu_C"], methods["Means"]["mu_N"]
        if mu_C > mu_N:
            p_C_est = (RM.mean() - mu_N) / (mu_C - mu_N)
        else:
            p_C_est = (mu_N - RM.mean()) / (mu_N - mu_C)

        # TODO: Check!
        # p_C_est = abs((RM.mean() - mu_N) / (mu_C - mu_N))
        results['Means'] = np.clip(p_C_est, 0.0, 1.0)

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
        # x_KDE = bins["centers"]
        # kde_mix = fit_kernel(Mix, bw=bins['width'], kernel=kernel)
        # res_mix = model.fit(np.exp(kde_mix.score_samples(x_KDE[:, np.newaxis])),
        #                     x=x_KDE, params=params_mix, method=method)
        # amp_R_C = res_mix.params['amp_1'].value
        # amp_R_N = res_mix.params['amp_2'].value
        # results['KDE'] = amp_R_C / (amp_R_C + amp_R_N)

    return results


def bootstrap_mixture(Mix, R_C, R_N, bins, methods, boot_size=-1, seed=None):
    """Generate a bootstrap of the mixture distribution and return an estimate
    of its proportion."""

    if boot_size == -1:
        boot_size = len(Mix)

    if seed is None:
        bs = np.random.choice(Mix, boot_size, replace=True)
    else:
        bs = np.random.RandomState(seed).choice(Mix, boot_size, replace=True)

    return point_estimate(bs, R_C, R_N, bins, methods)


def analyse_mixture(scores, bins='fd', methods='all', 
                    n_boot=1000, boot_size=-1, n_mix=0,
                    alpha=0.05, true_pC=None, ci_method="bca", correct_bias=False,
                    n_jobs=1, seed=None, verbose=1, logfile=''):
    """Analyse a mixture distribution and estimate the proportions of two
    reference distributions of which it is assumed to be comprised.

    Parameters
    ----------
    scores : dict
        A required dictionary of the form,
        `{‘R_C’: array_of_cases_scores,
          ‘R_N’: array_of_non-cases_scores,
          ‘Mix’: array_of_mix_scores}`.
    bins : str
        A string specifying the binning method:
        `['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']`.
        Default: `‘fd’`.
        Alternatively, a dictionary,
        `{‘width’: bin_width, ‘min’, min_edge, ‘max’: max_edge,
          ‘edges’: array_of_bin_edges, ‘centers’: array_of_bin_centers,
          ‘n’: number_of_bins}`.
    methods : str
        A string with the name of the method or `'all'` to run all methods
        (default). Alternatively, a list of method names (strings),
        `["Excess", "Means", "EMD", "KDE"]`, or a dictionary of (bool) flags,
        `{‘Excess’: True, ‘Means’: True, ‘EMD’: True, ‘KDE’: True}`.
    n_boot : int
        Number of bootstraps of the mixture to generate. Default: `1000`.
    boot_size : int
        The size of each mixture bootstrap. Default is the same size as the mixture.
    n_mix : int
        Number of mixtures to construct based on the initial point estimate.
        Default: `0`.
    alpha : float
        The alpha value for calculating confidence intervals from bootstrap
        distributions. Default: `0.05`.
    ci_method : str
        The name of the method used to calculate the confidence intervals.
        Defaults: `bca`.
    true_pC : float
        Optionally pass the true proportion of cases for comparing to the 
        estimated proportion(s).
    correct_bias : bool
        A boolean flag specifing whether to apply the bootstrap correction
        method or not. Default: `False`.
    n_jobs : int
        Number of bootstrap jobs to run in parallel. Default: `1`.
        Set `n_jobs = -1` runs on all CPUs.
    seed : int
        An optional value to seed the random number generator with for
        reproducibility (in the range [0, (2^32)-1]).
    verbose : int
        Integer to control the level of output (`0`, `1`, `2`). Set to `-1` to
        turn off all console output except the progress bars.
    logfile : str
        Optional filename for the output logs.
        Default: `"proportion_estimates.log"`.
 
    Returns
    -------
    df_pe : DataFrame
        A `pandas` dataframe of the proportion estimates. The first row is the
        point estimate. The remaining `n_boot * n_mix` rows are the
        bootstrapped estimates. Each column is the name of the estimation method.

    Alternatively if `correct_bias == True`:
    (df_pe, df_correct) : tuple
        A tuple of the proportion estimates and the corrected proportion
        estimates as dataframes.

    Additionally the logfile is written to the working directory.
    """


    if seed is not None:
        np.random.seed(seed)

    if correct_bias and n_mix + n_boot == 0:
        warnings.warn("No bootstraps - Ignoring bias correction!")

    # Boilerplate for backwards compatibility
    if "Ref1" in scores and "R_C" not in scores:
        scores["R_C"] = scores["Ref1"]  # Distribution of cases
    if "Ref2" in scores and "R_N" not in scores:
        scores["R_N"] = scores["Ref2"]  # Distribution of non-cases

    R_C = scores['R_C']
    R_N = scores['R_N']
    Mix = scores['Mix']

    if bins is None:
        bin_method = 'fd'
        hist, bins = estimate_bins(scores)
        bins = bins[bin_method]
    if isinstance(bins, str):
        bin_method = bins
        hist, bins = estimate_bins(scores)
        bins = bins[bin_method]
    
    if "method" in bins:
        bin_method = bins["method"]
    else:
        bin_method = "unknown"

    # Methods defaults to all if None is passed
    methods_ = prepare_methods(scores, bins, methods=methods, verbose=verbose)

    columns = [method for method in _ALL_METHODS_ if method in methods_]

    if logfile is not None:
        if logfile == '':
            logfile = "proportion_estimates.log"
        with open(logfile, 'w') as lf:
            lf.write("Distribution Summaries\n")
            lf.write("======================\n\n")
            lf.write(f"       | {'Mix':^7s} | {'R_C':^7s} | {'R_N':^7s} \n")
            lf.write("=====================================\n")
            lf.write(f"     n | {len(Mix):^7,} | {len(R_C):^7,} | {len(R_N):^7,} \n")
            lf.write(f"  Mean | {np.mean(Mix):^7.3} | {np.mean(R_C):^7.3f} | {np.mean(R_N):^7.3f} \n")
            lf.write(f"Median | {np.median(Mix):^7.3} | {np.median(R_C):^7.3f} | {np.median(R_N):^7.3f} \n")
            lf.write("\n\n")

    # Get initial estimate of proportions
    pe_initial = point_estimate(Mix, R_C, R_N, bins, methods_)
    if verbose > 1:
        print('Initial point estimates:')
        pprint(pe_initial)
    if true_pC:
        if verbose > 1:
            print(f"Ground truth: {true_pC:.5f}")
    # pe_initial = pd.DataFrame(pe_initial, index=[0], columns=columns)
    # pe_initial.to_dict(orient='records')  # Back to dictionary

    if n_boot > 0:
        if n_jobs == -1:
            nprocs = cpu_count()
        else:
            nprocs = n_jobs
        if verbose > 0:
            print(f'Running {n_boot} bootstraps with {nprocs} processors...',
                  flush=True)
            disable = False
        else:
            disable = True
        if verbose == -1:  # Allow only progress bar
            disable = False

        # Make bootstrapping deterministic with parallelism
        # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
        boot_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_boot)

        if n_mix <= 0:
            # HACK: This is to reduce the joblib overhead when n_jobs==1
            if n_jobs == 1 or n_jobs is None:
                # NOTE: These results are identical to when n_jobs==1 in the
                # parallel section however it takes about 25% less time per iteration
                results = [bootstrap_mixture(Mix, R_C, R_N, bins, methods_, boot_size, seed=None)
                           for b in trange(n_boot, desc="Bootstraps", dynamic_ncols=True, disable=disable)]
            else:
                with Parallel(n_jobs=n_jobs) as parallel:
                    results = parallel(delayed(bootstrap_mixture)(Mix, R_C, R_N, bins, methods_, boot_size, seed=b_seed)
                                       for b_seed in tqdm(boot_seeds, desc="Bootstraps", dynamic_ncols=True, disable=disable))
            # Put into dataframe
            pe_boot = pd.DataFrame.from_records(results, columns=columns)

        else:  # Extended mixture & bootstrap routine to calculate CIs
            # TODO: Refactor for efficiency
            sample_size = len(Mix)
            results = {}

            diable_method_bar = True
            diable_mix_bar = True
            disable_boot_bar = True
            if verbose > 0:
                diable_method_bar = False
                if verbose > 1:
                    diable_mix_bar = False
                    if verbose > 2:
                        disable_boot_bar = False
            if verbose == -1:
                diable_method_bar = False
                diable_mix_bar = False
                disable_boot_bar = False

            for method, prop_Ref1 in tqdm(pe_initial.items(), desc="Method",
                                          dynamic_ncols=True, disable=diable_method_bar):
                single_method = {}
                single_method[method] = methods_[method]
                mix_results = []

                for m in trange(n_mix, desc="Mixture", dynamic_ncols=True, disable=diable_mix_bar):

                    assert(0.0 <= prop_Ref1 <= 1.0)
                    n_C = int(round(sample_size * prop_Ref1))
                    n_N = sample_size - n_C

                    # Construct mixture
                    mixture = np.concatenate((np.random.choice(R_C, n_C, replace=True),
                                              np.random.choice(R_N, n_N, replace=True)))

                    # Spawn threads
                    with Parallel(n_jobs=nprocs) as parallel:
                        # Parallelise over mixtures
                        boot_list = parallel(delayed(bootstrap_mixture)(mixture, R_C, R_N, bins, single_method, boot_size, seed=b_seed)
                                             for b_seed in tqdm(boot_seeds,
                                                                desc="Bootstraps",
                                                                dynamic_ncols=True,
                                                                disable=disable_boot_bar))

                    mix_results.append(boot_list)

                # Concatenate each mixtures' bootstrap estimates
                # results[method] = [boot[method] for boot in boot_results
                #                    for boot_results in mix_results]
                results[method] = []
                for boot_results in mix_results:
                    for boot in boot_results:
                        results[method].append(boot[method])

            pe_boot = pd.DataFrame.from_records(results, columns=columns)

        # Put into dataframe
        pe_initial = pd.DataFrame(pe_initial, index=[0], columns=columns)
        df_pe = pd.concat([pe_initial, pe_boot], ignore_index=True)
        if n_mix > 0:
            index_arrays = list(itertools.product(range(1, n_mix+1), range(1, n_boot+1)))
            index_arrays.insert(0, (0, 0))  # Prepend 0, 0 for point estimate
            df_pe.index = pd.MultiIndex.from_tuples(index_arrays, names=["Remix", "Bootstrap"])

        if correct_bias:
            df_correct = correct_estimate(df_pe)
    else:
        df_pe = pd.DataFrame(pe_initial, index=[0], columns=columns)

    # ----------- Summarise proportions for the whole distribution ------------
    if verbose > 0 or logfile is not None:
        report = generate_report(df_pe, true_pC=true_pC, alpha=alpha, ci_method=ci_method)

    if verbose > 0:
        print("\n" + report + "\n")

    if logfile is not None:
        with open(logfile, 'a') as lf:
            lf.write(report)
            lf.write("\n")

    if correct_bias:
        return df_pe, df_correct
    return df_pe
