#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:06:16 2018

@author: ben
"""

# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
import time
from collections import defaultdict

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import scipy
from scipy.stats import gaussian_kde
from scipy.stats.distributions import norm

import statsmodels
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from datasets import load_diabetes_data


def plot_scaling(N=1000, bandwidth=0.1, rtol=0.0,
                 Nreps=3, kwds=None, xgrid=None):
    """
    Plot the time scaling of KDE algorithms.
    Either N, bandwidth, or rtol should be a 1D array.
    """
    if xgrid is None:
        xgrid = np.linspace(-10, 10, 5000)
    if kwds is None:
        kwds=dict()
    for name in functions:
        if name not in kwds:
            kwds[name] = {}
    times = defaultdict(list)
    
    B = np.broadcast(N, bandwidth, rtol)
    assert len(B.shape) == 1
    
    for N_i, bw_i, rtol_i in B:
        x = np.random.normal(size=N_i)
        kwds['Scikit-learn']['rtol'] = rtol_i
        for name, func in functions.items():
            t = 0.0
            for i in range(Nreps):
                t0 = time()
                func(x, xgrid, bw_i, **kwds[name])
                t1 = time()
                t += (t1 - t0)
            times[name].append(t / Nreps)
            
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw={'axisbg':'#EEEEEE',
                                       'axisbelow':True})
    ax.grid(color='white', linestyle='-', linewidth=2)
    plot_kwds={'linewidth':3, 'alpha':0.5}
    
    if np.size(N) > 1:
        for name in kde_funcnames:
            ax.loglog(N, times[name], label=name, **plot_kwds)
        ax.set_xlabel('Number of points')
    elif np.size(bandwidth) > 1:
        for name in kde_funcnames:
            ax.loglog(bandwidth, times[name], label=name, **plot_kwds)
        ax.set_xlabel('Bandwidth')
    elif np.size(rtol) > 1:
        for name in kde_funcnames:
            ax.loglog(rtol, times[name], label=name, **plot_kwds)
        ax.set_xlabel('Relative Tolerance')
        
    for spine in ax.spines.values():
        spine.set_color('#BBBBBB')
    ax.legend(loc=0)
    ax.set_ylabel('time (seconds)')
    ax.set_title('Execution time for KDE '
                 '({0} evaluations)'.format(len(xgrid)))
    
    return times


def plot_scaling_vs_kernel(kernels, N=1000, bandwidth=0.1, rtol=0.0,
                           Nreps=3, kwds=None, xgrid=None):
    """
    Plot the time scaling for Scikit-learn kernels.
    Either N, bandwidth, or rtol should be a 1D array.
    """
    if xgrid is None:
        xgrid = np.linspace(-10, 10, 5000)
    if kwds is None:
        kwds=dict()
    times = defaultdict(list)
    
    B = np.broadcast(N, bandwidth, rtol)
    assert len(B.shape) == 1
    
    for N_i, bw_i, rtol_i in B:
        x = np.random.normal(size=N_i)
        for kernel in kernels:
            kwds['kernel'] = kernel
            kwds['rtol'] = rtol_i
            t = 0.0
            for i in range(Nreps):
                t0 = time()
                kde_sklearn(x, xgrid, bw_i, **kwds)
                t1 = time()
                t += (t1 - t0)
            times[kernel].append(t / Nreps)
            
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw={'axisbg':'#EEEEEE',
                                       'axisbelow':True})
    ax.grid(color='white', linestyle='-', linewidth=2)
    plot_kwds={'linewidth':3, 'alpha':0.5}
    
    if np.size(N) > 1:
        for kernel in kernels:
            ax.loglog(N, times[kernel], label=kernel, **plot_kwds)
        ax.set_xlabel('Number of points')
    elif np.size(bandwidth) > 1:
        for kernel in kernels:
            ax.loglog(bandwidth, times[kernel], label=kernel, **plot_kwds)
        ax.set_xlabel('Bandwidth')
    elif np.size(rtol) > 1:
        for kernel in kernels:
            ax.loglog(rtol, times[kernel], label=kernel, **plot_kwds)
        ax.set_xlabel('Relative Tolerance')
        
    for spine in ax.spines.values():
        spine.set_color('#BBBBBB')
    ax.legend(loc=0)
    ax.set_ylabel('time (seconds)')
    ax.set_title('Execution time for KDE '
                 '({0} evaluations)'.format(len(xgrid)))
    
    return times


def plot_kernels():
    """Visualize the KDE kernels available in Scikit-learn"""
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw={'axisbg':'#EEEEEE',
                                       'axisbelow':True})
    ax.grid(color='white', linestyle='-', linewidth=2)
    for spine in ax.spines.values():
        spine.set_color('#BBBBBB')

    X_src = np.zeros((1, 1))
    x_grid = np.linspace(-3, 3, 1000)

    for kernel in ['gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine']:
        log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(x_grid[:, None])
        ax.plot(x_grid, np.exp(log_dens), lw=3, alpha=0.5, label=kernel)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2.9, 2.9)
    ax.legend()


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def plot_kernels():
    """Visualize the KDE kernels available in Scikit-learn"""
    fig, ax = plt.subplots(figsize=(8, 6),
                           subplot_kw={'axisbg':'#EEEEEE',
                                       'axisbelow':True})
    ax.grid(color='white', linestyle='-', linewidth=2)
    for spine in ax.spines.values():
        spine.set_color('#BBBBBB')

    X_src = np.zeros((1, 1))
    x_grid = np.linspace(-3, 3, 1000)

    for kernel in ['gaussian', 'tophat', 'epanechnikov',
                   'exponential', 'linear', 'cosine']:
        log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(x_grid[:, None])
        ax.plot(x_grid, np.exp(log_dens), lw=3, alpha=0.5, label=kernel)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2.9, 2.9)
    ax.legend()

#kde_funcs = [kde_statsmodels_u, kde_statsmodels_m, kde_scipy, kde_sklearn]
#kde_funcnames = ['Statsmodels-U', 'Statsmodels-M', 'Scipy', 'Scikit-learn']

kde_funcs = [kde_statsmodels_u, kde_scipy, kde_sklearn]
kde_funcnames = ['Statsmodels-U', 'Scipy', 'Scikit-learn']
functions = dict(zip(kde_funcnames, kde_funcs))

print("Package Versions:")
print("  scikit-learn:", sklearn.__version__)
print("  scipy:", scipy.__version__)
print("  statsmodels:", statsmodels.__version__)
print()

# Plot sklearn kernels
#plot_kernels()

scores, bins, means, medians, p_C = load_diabetes_data("T1GRS")

# The grid we'll use for plotting
#x_grid = np.linspace(-4.5, 3.5, 1000)
x_grid = bins["centers"]
x_grid = np.linspace(bins["min"], bins["max"], 100)

# Draw points from a bimodal distribution in 1D
np.random.seed(0)
#x = np.concatenate([norm(-1, 1.).rvs(400),
#                    norm(1, 0.3).rvs(100)])
#pdf_true = (0.8 * norm(-1, 1).pdf(x_grid) +
#            0.2 * norm(1, 0.3).pdf(x_grid))

fig, ax = plt.subplots(len(scores), len(kde_funcs), sharey=True, sharex=True,
                       figsize=(12, 9))
for r, (label, data) in enumerate(scores.items()):
    x = data  # scores["R_C"]


#    grid = GridSearchCV(KernelDensity(),
#                        {'bandwidth': np.logspace(-2, 0, 9)},
#                        cv=20) # 20-fold cross-validation
#    grid.fit(x[:, None])
#    print(grid.best_params_)

    # Plot the three kernel density estimates
#    fig, ax = plt.subplots(1, len(kde_funcs), sharey=True,
#                           figsize=(12, 3))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(len(kde_funcs)):
        t = time.time()  # Start timer
        pdf = kde_funcs[i](x, x_grid, bandwidth=0.05)
        elapsed = time.time() - t
        print('{}: {} elapsed time = {:.3f} s'.format(label, kde_funcnames[i], elapsed))

        ax[r, i].plot(x_grid, pdf, color='red', alpha=0.5, lw=3)
    #    ax[i].fill(x_grid, pdf_true, ec='gray', fc='gray', alpha=0.4)
        ax[r, i].hist(x, bins["edges"], density=True)
        if r == 0:
            ax[r, i].set_title(kde_funcnames[i] + "; N={}".format(len(x_grid)))
    #    ax[i].set_xlim(-4.5, 3.5)
    ax[r, 0].set_ylabel(label + "; N={}".format(len(data)))
    print()
#plt.tight_layout()

# sklearn
#kde = KernelDensity(kernel=KDE_kernel, bandwidth=0.1).fit(np.random.choice(scores['R_C'], size=30000)[:, np.newaxis])
#kde.score_samples(scores['Mix'][:, np.newaxis])


# Compare bandwitch calculation methods
# Compute on the two reference populations
x = np.r_[scores["R_C"], scores["R_N"]]


plt.figure(figsize=(12, 9))
plt.hist(x, bins["edges"], density=True)

grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.logspace(-2, 0, 17)},
                    cv=12) # 20-fold cross-validation
grid.fit(x[:, None])
bw = grid.best_params_["bandwidth"]
pdf = kde_sklearn(x, x_grid, bandwidth=bw)
plt.plot(x_grid, pdf, color='red', alpha=0.5, lw=3, label="sklearn: bw={}".format(bw))
print("Scikit-learn CV", bw)

kde = gaussian_kde(x, bw_method='scott')
kde.evaluate(x)
plt.plot(x_grid, kde.evaluate(x_grid), color='blue', alpha=0.5, lw=3, label="Scipy Scott: bw={}".format(kde.factor))
print("Scipy: Scott", kde.factor)

kde = gaussian_kde(x, bw_method='silverman')
kde.evaluate(x)
plt.plot(x_grid, kde.evaluate(x_grid), color='green', alpha=0.5, lw=3, label="Scipy Silverman: bw={}".format(kde.factor))
print("Scipy: Silverman", kde.factor)

plt.legend()
plt.savefig("kde_bw_estimation_comparison.png")

plot_kernels()


# Scaling with the Number of Points
plot_scaling(N=np.logspace(1, 4, 10),
             kwds={'Statsmodels-U':{'fft':False}})

plot_scaling(N=np.logspace(1, 4, 10),
             rtol=1E-4,
             kwds={'Statsmodels-U':{'fft':True}})

# Dependence on rtol
plot_scaling(N=1E4,
             rtol=np.logspace(-16, -1, 10),
             bandwidth=0.2)

# Dependence on Bandwidth
plot_scaling(N=1E4, rtol=1E-4,
             bandwidth=np.logspace(-4, 3, 10))

# Dependence on Kernel
plot_scaling_vs_kernel(kernels=['tophat', 'linear', 'exponential', 'gaussian'],
                       bandwidth=np.logspace(-4, 3, 10),
                       N=1E4, rtol=1E-4)

plot_scaling_vs_kernel(kernels=['tophat', 'linear', 'exponential', 'gaussian'],
                       bandwidth=0.15, N=1E4, rtol=np.logspace(-16, -1, 10))

plot_scaling_vs_kernel(kernels=['tophat', 'linear', 'exponential', 'gaussian'],
                       bandwidth=0.15, rtol=1E-4, N=np.logspace(1, 4, 10))

# TODO: Try other KDE functions e.g.:
#x = np.random.normal(0, 1, size=30)
#bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
#support = np.linspace(-4, 4, 200)
#
#kernels = []
#for x_i in x:
#
#    kernel = sp.stats.norm(x_i, bandwidth).pdf(support)
#    kernels.append(kernel)
#    plt.plot(support, kernel, color="r")
#
#sns.rugplot(x, color=".2", linewidth=3);


