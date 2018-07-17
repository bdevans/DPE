#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:35:23 2018

@author: ben
"""
import time
from pprint import pprint

from scipy.optimize import curve_fit

from lmfit import minimize

import corner


class Timer:
    """
    Class for timing blocks of code.
    http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
    Examples
    --------
    >>> with Timer() as t:
    >>>    run code...
    Execution took <t>s)
    """
    # interval = 0
    def __init__(self):
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('{:.3g}s'.format(self.interval))

    def __str__(self):
        return '{:.3g}s'.format(self.interval)

    def reset(self):
        """Reset timer to 0."""
        self.start = 0
        self.end = 0
        self.interval = 0

# ------------------------------ KDE method ------------------------------

plot_results = True
verbose = 1

KDE_kernel = "gaussian"
bw = bins["width"]
print("Using {} kernel with bandwith = {}".format(KDE_kernel, bw))

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


# SciPy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
print("="*40)
print("SciPy curve_fit")
print("="*40)

def KDE_Mix(x, amp_Ref1, amp_Ref2):
    '''The model function, f(x, …). It must take the independent variable as the
    first argument and the parameters to fit as separate remaining arguments.'''
    dens_Ref1 = np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))
    dens_Ref2 = np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))
    return amp_Ref1 * dens_Ref1 + amp_Ref2 * dens_Ref2

p0 = (1, 1)

with Timer() as t:
    popt, pcov = curve_fit(KDE_Mix, x, y, p0)

print("Parameters:", popt)
print("Covariance:", pcov)
amp_Ref1, amp_Ref2 = popt
print("Estimated proportion:", amp_Ref1/(amp_Ref1+amp_Ref2))
print()


# lmfit model
print("="*40)
print("LMFIT: model")
print("="*40)

#def fit_KDE_model(Mix, bins, model, params_mix, kernel):
#    x_KDE = np.linspace(bins['min'], bins['max'], len(Mix)+2)
#    mix_kde = KernelDensity(kernel=kernel, bandwidth=bins['width']).fit(Mix[:, np.newaxis])
#    res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
#    amp_Ref1 = res_mix.params['amp_Ref1'].value
#    amp_Ref2 = res_mix.params['amp_Ref2'].value
#    return amp_Ref1/(amp_Ref1+amp_Ref2)


# Define the KDE models
# x := Bin centres originally with n_bins = int(np.floor(np.sqrt(N)))
def kde_Ref1(x, amp_Ref1):
    return amp_Ref1 * np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))

def kde_Ref2(x, amp_Ref2):
    return amp_Ref2 * np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))

model = lmfit.Model(kde_Ref1) + lmfit.Model(kde_Ref2)

params_mix = model.make_params()
params_mix['amp_Ref1'].value = 1
params_mix['amp_Ref1'].min = 0
params_mix['amp_Ref2'].value = 1
params_mix['amp_Ref2'].min = 0

with Timer() as t:
    res_model = model.fit(y, x=x, params=params_mix)

print("Parameters:")
res_model.params.pretty_print()
amp_Ref1 = res_model.params['amp_Ref1'].value
amp_Ref2 = res_model.params['amp_Ref2'].value
print("Estimated proportion:", amp_Ref1/(amp_Ref1+amp_Ref2))
print()





dely = res_model.eval_uncertainty(sigma=3)

amp_T1 = res_model.params['amp_Ref1'].value
amp_T2 = res_model.params['amp_Ref2'].value

kde1 = kde_Ref1(x, amp_T1)
kde2 = kde_Ref2(x, amp_T2)


if plot_results:
    plt.figure()
    fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False)

    axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=["Reference 1", "Reference 2"])
    legend = axP.legend(facecolor='grey')
    #legend.get_frame().set_facecolor('grey')
    axP.set_title('Proportions of Ref 1 and Ref 2')

    plt.sca(axM)
    res_model.plot_fit()

    axM.fill_between(x, res_model.best_fit-dely, res_model.best_fit+dely, color="#ABABAB")

    plt.sca(axR)
    res_model.plot_residuals()

    #plt.sca(axI)
    axI.plot(x, kde1, label='Reference 1')
    axI.plot(x, kde2, label='Reference 2')
    axI.legend()

if verbose:
    print(res_model.fit_report())
    print('Ref2/Ref1 =', amp_Ref2/amp_Ref1)
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
    kde_Ref1 = np.exp(kdes['Ref1'][KDE_kernel].score_samples(x[:, np.newaxis]))
    kde_Ref2 = np.exp(kdes['Ref2'][KDE_kernel].score_samples(x[:, np.newaxis]))
    model = pars['amp_Ref1'].value * kde_Ref1 + pars['amp_Ref2'].value * kde_Ref2
    if data is None:
        return model
    else:
        return model - data


with Timer() as t:
    res_min = minimize(KDE_model, params_mix, args=(x, kdes, y), method=method)

print("Parameters:")
res_min.params.pretty_print()
amp_Ref1 = res_min.params['amp_Ref1'].value
amp_Ref2 = res_min.params['amp_Ref2'].value
print("Estimated proportion:", amp_Ref1/(amp_Ref1+amp_Ref2))
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
amp_Ref1 = res_mo.params['amp_Ref1'].value
amp_Ref2 = res_mo.params['amp_Ref2'].value
print("Estimated proportion:", amp_Ref1/(amp_Ref1+amp_Ref2))
print(lmfit.fit_report(res_mo.params))
lmfit.printfuncs.report_fit(res_mo)

ci = lmfit.conf_interval(mini, res_mo)
lmfit.printfuncs.report_ci(ci)
print()


if False:
    # Calculating the posterior probability distribution of parameters
    res = mini.emcee(burn=300, steps=1000, thin=20, params=mi.params)
    corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()))
