#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:35:23 2018

@author: ben
"""
# Script to benchmark performance of the KDE method
# There are two main computational costs
# 1. Convolving kernels with the data. 
# This depends upon: the kernel, bandwidth, number of points and the algorithm
# 2. The least-squares fitting algorithm

from pprint import pprint

import numpy as np
from scipy.optimize import curve_fit
# from lmfit import minimize
import lmfit
import corner
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt

from datasets import load_diabetes_data, load_coeliac_data
from proportion_estimation import analyse_mixture
from proportion_estimation import fit_kernel
from utilities import Timer

# scores, bins, means, medians, p_C = load_diabetes_data('T1GRS')
scores, bins, means, medians, p_C = load_coeliac_data()

# seed = 42
# n_boot = 10
# sample_size = 1000  # -1
# n_mix = 10
# alpha = 0.05
# methods = 'all'
# n_jobs = 1
# ci_method = "bca"
# correct_bias = False
# kernel = "gaussian"
# n_head = 50

# ------------------------------ KDE method ------------------------------

plot_results = True
verbose = 0

KDE_kernel = "gaussian"
bw = bins["width"]
print(f"Using {KDE_kernel} kernel with bandwith = {bw}")

# TODO: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

# Set up
def fit_kernels(scores, bw):
    kernels = {}
    for label, data in scores.items():
        kernels[label] = {}
        X = data[:, np.newaxis]
        for kernel in ['gaussian', 'tophat', 'epanechnikov',
                       'exponential', 'linear', 'cosine']:
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)  # , atol=1e-2, rtol=1e-2
            kernels[label][kernel] = kde
    return kernels

# Fit all kernels for each distribution
kdes = fit_kernels(scores, bw)

x = bins["centers"]
y = np.exp(kdes["Mix"][KDE_kernel].score_samples(x[:, np.newaxis]))


# SciPy. This defaults to method 'lm'
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
print("="*40)
print("SciPy curve_fit")
print("="*40)

def KDE_Mix(x, amp_R_C, amp_R_N):
    '''The model function, f(x, ...). It must take the independent variable as the
    first argument and the parameters to fit as separate remaining arguments.'''
    dens_R_C = np.exp(kdes['R_C'][KDE_kernel].score_samples(x[:, np.newaxis]))
    dens_R_N = np.exp(kdes['R_N'][KDE_kernel].score_samples(x[:, np.newaxis]))
    return amp_R_C * dens_R_C + amp_R_N * dens_R_N

p0 = (1, 1)

with Timer() as t:
    popt, pcov = curve_fit(KDE_Mix, x, y, p0)

print("Parameters:", popt)
print("Covariance:", pcov)
amp_R_C, amp_R_N = popt
print("Estimated proportion:", amp_R_C/(amp_R_C+amp_R_N))
print()


# SciPy with bounds. This uses method 'trf'.
# Internally, this calls least_squares instead of leastsq
print("="*40)
print("SciPy curve_fit (with bounds)")
print("="*40)

p0 = (1, 1)
# bounds = (0, 1)  # 2-tuple of arrays (separate bounds for each param) or scalars
bounds = (0, np.inf)

with Timer() as t:
    popt, pcov = curve_fit(KDE_Mix, x, y, p0, bounds=bounds)

print("Parameters:", popt)
print("Covariance:", pcov)
amp_R_C, amp_R_N = popt
print("Estimated proportion:", amp_R_C/(amp_R_C+amp_R_N))
print()

# lmfit model
print("="*40)
print("LMFIT: model")
print("="*40)

#def fit_KDE_model(Mix, bins, model, params_mix, kernel):
#    x_KDE = np.linspace(bins['min'], bins['max'], len(Mix)+2)
#    mix_kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(Mix[:, np.newaxis])
#    res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
#    amp_R_C = res_mix.params['amp_R_C'].value
#    amp_R_N = res_mix.params['amp_R_N'].value
#    return amp_R_C/(amp_R_C+amp_R_N)


# Define the KDE models
# x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
def kde_R_C(x, amp_R_C):
    return amp_R_C * np.exp(kdes['R_C'][KDE_kernel].score_samples(x[:, np.newaxis]))

def kde_R_N(x, amp_R_N):
    return amp_R_N * np.exp(kdes['R_N'][KDE_kernel].score_samples(x[:, np.newaxis]))

model = lmfit.Model(kde_R_C) + lmfit.Model(kde_R_N)

params_mix = model.make_params()
params_mix['amp_R_C'].value = 1
params_mix['amp_R_C'].min = 0
params_mix['amp_R_N'].value = 1
params_mix['amp_R_N'].min = 0

with Timer() as t:
    res_model = model.fit(y, x=x, params=params_mix)

print("Parameters:")
res_model.params.pretty_print()
amp_R_C = res_model.params['amp_R_C'].value
amp_R_N = res_model.params['amp_R_N'].value
print("Estimated proportion:", amp_R_C/(amp_R_C+amp_R_N))
print()



if plot_results:
    dely = res_model.eval_uncertainty(sigma=3)

    amp_T1 = res_model.params['amp_R_C'].value
    amp_T2 = res_model.params['amp_R_N'].value

    kde1 = kde_R_C(x, amp_T1)
    kde2 = kde_R_N(x, amp_T2)

    # plt.figure()
    fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(12, 8))

    axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=["$R_C$", "$R_N$"])
    legend = axP.legend(facecolor='grey')
    #legend.get_frame().set_facecolor('grey')
    axP.set_title('Proportions of R_C and R_N')

    plt.sca(axM)
    res_model.plot_fit()

    axM.fill_between(x, res_model.best_fit-dely, res_model.best_fit+dely, color="#ABABAB")

    plt.sca(axR)
    res_model.plot_residuals()

    #plt.sca(axI)
    axI.plot(x, kde1, label='$R_C$')
    axI.plot(x, kde2, label='$R_N$')
    axI.legend()
    plt.savefig("fitting_comparison.png")

if verbose:
    print(res_model.fit_report())
    print('R_C/R_N =', amp_R_C/amp_R_N)
    print('')
    print('\nParameter confidence intervals:')
    print(res_model.ci_report())  # --> res_mix.ci_out # See also res_mix.conf_interval()





# TODO: Print out warnings if goodness of fit is poor?


# lmfit minimize
# https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize
method = 'leastsq'

print("="*40)
print("LMFIT: minimize")
print("="*40)


# TODO: Switch to using Minimizer to avoid issues with lmfit models
def KDE_model(pars, x, kdes, data=None):
    kde_R_C = np.exp(kdes['R_C'][KDE_kernel].score_samples(x[:, np.newaxis]))
    kde_R_N = np.exp(kdes['R_N'][KDE_kernel].score_samples(x[:, np.newaxis]))
    model = pars['amp_R_C'].value * kde_R_C + pars['amp_R_N'].value * kde_R_N
    if data is None:
        return model
    else:
        return model - data


with Timer() as t:
    res_min = lmfit.minimize(KDE_model, params_mix, args=(x, kdes, y), method=method)

print("Parameters:")
res_min.params.pretty_print()
amp_R_C = res_min.params['amp_R_C'].value
amp_R_N = res_min.params['amp_R_N'].value
print("Estimated proportion:", amp_R_C/(amp_R_C+amp_R_N))
print()



# For ful control there is a Minimiser class
# https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer
print("="*40)
print("LMFIT: Minimizer")
print("="*40)

kws = {'x': x, 'kdes': kdes, 'data': y}
mini = lmfit.Minimizer(KDE_model, params_mix, fcn_args=(x, kdes, y))
with Timer() as t:
    res_mo = mini.minimize(method=method)

print("Parameters:")
res_mo.params.pretty_print()
amp_R_C = res_mo.params['amp_R_C'].value
amp_R_N = res_mo.params['amp_R_N'].value
print("Estimated proportion:", amp_R_C/(amp_R_C+amp_R_N))
if verbose:
    print(lmfit.fit_report(res_mo.params))
    lmfit.printfuncs.report_fit(res_mo)

    ci = lmfit.conf_interval(mini, res_mo)
    lmfit.printfuncs.report_ci(ci)
    print()


if False:
    # Calculating the posterior probability distribution of parameters
    res = mini.emcee(burn=300, steps=1000, thin=20, params=mini.params)
    corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
