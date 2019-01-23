#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:47:27 2018

Module to analyse an unknown mixture population.

@author: ben
"""

# TODO: Rename {Ref1: R_D, Ref2: R_N}
# TODO: Move point_estimate and bootstrap_mixture to top level and hide other functions

from pprint import pprint
import itertools
# import warnings

import numpy as np
import scipy as sp
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


def fit_kernels(scores, bw, kernel='gaussian'):
    """No longer used."""
    kernels = {}
    for label, data in scores.items():
        X = data[:, np.newaxis]
        kernels[label] = KernelDensity(kernel=kernel, bandwidth=bw,
                                       atol=0, rtol=1e-4).fit(X)
    return kernels


def fit_kernel(scores, bw, kernel='gaussian'):
    X = scores[:, np.newaxis]
    return KernelDensity(kernel=kernel, bandwidth=bw, atol=0, rtol=1e-4).fit(X)


# @mem.cache
def fit_KDE_model(Mix, bins, model, params_mix, kernel, method='leastsq'):
    # model = methods["model"]
    x_KDE = bins["centers"]
    kde_mix = KernelDensity(kernel=kernel, bandwidth=bins['width'])
    kde_mix.fit(Mix[:, np.newaxis])
    res_mix = model.fit(np.exp(kde_mix.score_samples(x_KDE[:, np.newaxis])),
                        x=x_KDE, params=params_mix, method=method)
    amp_Ref1 = res_mix.params['amp_1'].value
    amp_Ref2 = res_mix.params['amp_2'].value
    return amp_Ref1 / (amp_Ref1 + amp_Ref2)


def interpolate_CDF(scores, x_i, min_edge, max_edge):
    # TODO: x = [x_i[0], *sorted(scores), x_i[-1]]
    x = [min_edge, *sorted(scores), max_edge]
    y = np.linspace(0, 1, num=len(x), endpoint=True)
    (iv, ii) = np.unique(x, return_index=True)
    return np.interp(x_i, iv, y[ii])


def prepare_methods(scores, bins, methods=None, verbose=1):

    if methods is None:  # Run all methods
        methods = {method: True for method in _ALL_METHODS_}

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
            # kdes = fit_kernels(scores, methods["KDE"]["bandwidth"], methods["KDE"]["kernel"])
            # kde_1 = kdes["Ref1"]  # [methods["KDE"]["kernel"]]
            # kde_2 = kdes["Ref2"]  # [methods["KDE"]["kernel"]]
            kde_1 = fit_kernel(scores["Ref1"], methods["KDE"]["bandwidth"],
                               methods["KDE"]["kernel"])
            kde_2 = fit_kernel(scores["Ref2"], methods["KDE"]["bandwidth"],
                               methods["KDE"]["kernel"])

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

    return methods


def calc_conf_intervals(values, initial=None, average=np.mean, alpha=0.05, ci_method="experimental"):

    n_obs = len(values)
    average_value = average(values)

    if ci_method == 'experimental':
        assert initial is not None and 0.0 <= initial <= 1.0
        err_low, err_upp = np.percentile(values-initial, [100*alpha/2, 100*(1-alpha/2)])
        ci_low, ci_upp = initial-err_upp, initial-err_low
    elif ci_method == 'centile':
        ci_low, ci_upp = np.percentile(values, [100*alpha/2, 100*(1-alpha/2)])
    elif ci_method == 'stderr':
        p = average_value
        # NOTE: This currently allows CIs outside [0, 1]
        err = np.sqrt(p*(1-p)/n_obs) * sp.stats.norm.ppf(1-alpha/2)
        ci_low, ci_upp = p - err, p + err
    else:  # Assumes a binomial distribution
        count = int(average_value*n_obs)
        ci_low, ci_upp = proportion_confint(count, n_obs, alpha=alpha,
                                            method=ci_method)

    return ci_low, ci_upp


def generate_report(df_pe, true_p1=None, alpha=0.05, ci_method="experimental"):
    # TODO: Incorporate ipoint estimates in report
    pe_point = df_pe.iloc[0, :]
    pe_boot = df_pe.iloc[1:, :]
    n_boot = len(pe_boot)  # len(df_pe)
    line_width = 54
    report = []
    report.append(" {:^12} | {:^17s} | {:^17s} ".format("Method",
                                                        "Estimated pC",   # Reference 1
                                                        "Estimated pN"))  # Reference 2
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
            if ci_method == "experimental":
                bias = point_est - np.mean(values)
                corrected_est = point_est + bias
                report.append(" Corrected    | {:<17.5f} | {:<17.5f} "
                              .format(corrected_est, 1-corrected_est))
        report.append("-"*line_width)
    if true_p1:
        report.append(" {:12} | {:<17.5f} | {:<17.5f} "
                      .format("Ground Truth", true_p1, 1-true_p1))
        report.append("="*line_width)
    # report.append("\n")
    return "\n".join(report)


def point_estimate(RM, Ref1, Ref2, bins, methods=None):
    '''Estimate the proportion of two reference populations comprising
    an unknown mixture.

    The returned proportions are with respect to Ref_1, the disease group.
    The proportion of Ref_2, p_2, is assumed to be 1 - p_1.
    '''

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


def bootstrap_mixture(Mix, Ref1, Ref2, bins, methods, boot_size=-1, seed=None):

    if boot_size == -1:
        boot_size = len(Mix)

    if seed is None:
        bs = np.random.choice(Mix, boot_size, replace=True)
    else:
        bs = np.random.RandomState(seed).choice(Mix, boot_size, replace=True)

    return point_estimate(bs, Ref1, Ref2, bins, methods)


def analyse_mixture(scores, bins, methods, n_boot=1000, boot_size=-1, n_mix=0,
                    alpha=0.05, true_p1=None, n_jobs=1, seed=None,
                    verbose=1, logfile=''):

    if seed is not None:
        np.random.seed(seed)

    Ref1 = scores['Ref1']
    Ref2 = scores['Ref2']
    Mix = scores['Mix']

    methods = prepare_methods(scores, bins, methods=methods, verbose=verbose)

    columns = [method for method in _ALL_METHODS_ if method in methods]

    # Get initial estimate of proportions
    pe_initial = point_estimate(Mix, Ref1, Ref2, bins, methods)
    if verbose > 1:
        pprint(pe_initial)
    if true_p1:
        if verbose > 1:
            print("Ground truth: {:.5f}".format(true_p1))
    # pe_initial = pd.DataFrame(pe_initial, index=[0], columns=columns)
    # pe_initial.to_dict(orient='records')  # Back to dictionary

    if n_boot > 0:
        if n_jobs == -1:
            nprocs = cpu_count()
        else:
            nprocs = n_jobs
        if verbose > 0:
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

        if n_mix <= 0:
            # HACK: This is to reduce the joblib overhead when n_jobs==1
            if n_jobs == 1 or n_jobs is None:
                # NOTE: These results are identical to when n_jobs==1 in the
                # parallel section however it takes about 25% less time per iteration
                results = [bootstrap_mixture(Mix, Ref1, Ref2, bins, methods, boot_size, seed=None)
                           for b in trange(n_boot, desc="Bootstraps", dynamic_ncols=True, disable=disable)]
            else:
                with Parallel(n_jobs=n_jobs) as parallel:
                    results = parallel(delayed(bootstrap_mixture)(Mix, Ref1, Ref2, bins, methods, boot_size, seed=b_seed)
                                       for b_seed in tqdm(boot_seeds, desc="Bootstraps", dynamic_ncols=True, disable=disable))
            # Put into dataframe
            pe_boot = pd.DataFrame.from_records(results, columns=columns)

        else:  # Extended mixture & bootstrap routine to calculate CIs
            # TODO: Refactor for efficiency
            sample_size = len(Mix)
            results = {}

            for method, prop_Ref1 in tqdm(pe_initial.items(), desc="Method",
                                          dynamic_ncols=True, disable=disable):
                single_method = {}
                single_method[method] = methods[method]
                mix_results = []

                for m in trange(n_mix, desc="Mixture", dynamic_ncols=True, disable=disable):

                    assert(0.0 <= prop_Ref1 <= 1.0)
                    n_Ref1 = int(round(sample_size * prop_Ref1))
                    n_Ref2 = sample_size - n_Ref1

                    # Construct mixture
                    mixture = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                                              np.random.choice(Ref2, n_Ref2, replace=True)))

                    # Spawn threads
                    with Parallel(n_jobs=nprocs) as parallel:
                        # Parallelise over mixtures
                        boot_list = parallel(delayed(bootstrap_mixture)(mixture, Ref1, Ref2, bins, single_method, boot_size, seed=b_seed)
                                             for b_seed in tqdm(boot_seeds,
                                                                desc="Bootstraps",
                                                                dynamic_ncols=True,
                                                                disable=disable))

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
    else:
        df_pe = pd.DataFrame(pe_initial, index=[0], columns=columns)

    # ----------- Summarise proportions for the whole distribution ------------
    if verbose > 0 or logfile is not None:
        report = generate_report(df_pe, true_p1=true_p1, alpha=alpha)

    if verbose > 0:
        print("\n" + report + "\n")

    if logfile is not None:
        if logfile == '':
            logfile = "proportion_estimates.log"
        with open(logfile, 'w') as lf:
            lf.write(report)
            lf.write("\n")

    return df_pe
