#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:25:30 2017

@author: ben
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
import scipy as sp
#from scipy import stats
#from fitter import Fitter
from sklearn.neighbors import KernelDensity
import lmfit


#sns.color_palette('pastel')
mpl.style.use('seaborn')
mpl.rc('figure', figsize=(12, 10))
np.seterr(divide='ignore', invalid='ignore')

#xls = pd.ExcelFile("data.xls")
#data = xls.parse()
data = pd.read_csv('data.csv', usecols=[1,2])
data.rename(columns={'diabetes_type': 'type', 't1GRS': 'T1GRS'}, inplace=True)
data.describe()

T1 = data.loc[data['type'] == 1, 'T1GRS'].as_matrix()
T2 = data.loc[data['type'] == 2, 'T1GRS'].as_matrix()
Mix = data.loc[data['type'] == 3, 'T1GRS'].as_matrix()

#------------------------------ Bin the data ---------------------------------
N = data.count()[0]
n_bins = int(np.floor(np.sqrt(N)))

#(freqs, bins, patches) = plt.hist([T1, T2, Mix], n_bins, histtype='barstacked')
#plt.legend()

#(freqs_T1, freqs_T2, freqs_Mix) = freqs

#bin_width = 0.005
#bin_edges = np.arange(0.095, 0.35+bin_width, bin_width)
#bin_centers = np.arange(0.095+bin_width/2, 0.35+bin_width/2, bin_width)

(freqs_T1, bins) = np.histogram(T1, bins=n_bins)
(freqs_T2, bins) = np.histogram(T2, bins=n_bins)
(freqs_Mix, bins) = np.histogram(Mix, bins=n_bins)

x = (bins[:-1] + bins[1:]) / 2  # Bin centres
y = Mix

#-----------------------------------------------------------------------------


if False:
    ## Fit to Gaussians
    
    
    #fT1.fit()
    #fT1 = Fitter(freqs_T1)
    #fT1.summary()
    
    def gaussian(x, amp1, cen1, wid1):
        "Mixture of two 1-D Gaussians"
    
        gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-cen1)**2 /(2*wid1**2))
        return gauss1
    
    
    def gaussian_mixture(x, amp1, cen1, wid1, amp2, cen2, wid2):
        "Mixture of two 1-D Gaussians"
    
        gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-cen1)**2 /(2*wid1**2))
        gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-cen2)**2 /(2*wid2**2))
        return gauss1 + gauss2
    
    #mix = lmfit.Model(gaussian_mixture)
    #y = data.loc[data['type'] == 3, 'T1GRS'].as_matrix()
    
    #res = mix.fit(data[data['type'] == 3])
    
    #gauss = lmfit.Model(gaussian)
    #res = gauss.fit(freqs_T1)
        
    
    model_T1 = lmfit.models.GaussianModel(prefix='T1')
    res_T1 = model_T1.fit(freqs_T1, x=x)
    print(res_T1.fit_report())
    
    plt.figure()
    fig, (ax1, ax2, axM) = plt.subplots(3, 1, sharex=True, sharey=True)
    #ax1.plot(x, freqs_T1, 'o')
    #ax1.plot(x, res_T1.best_fit, 'k-')
    plt.sca(ax1)
    res_T1.plot_fit()
    dely = res_T1.eval_uncertainty(sigma=3)
    ax1.fill_between(x, res_T1.best_fit-dely, res_T1.best_fit+dely, color="#ABABAB")
    
    
    model_T2 = lmfit.models.GaussianModel(prefix='T2')
    res_T2 = model_T2.fit(freqs_T2, x=x)
    print(res_T2.fit_report())
    
    
    plt.sca(ax2)
    res_T2.plot_fit()
    dely = res_T2.eval_uncertainty(sigma=3)
    ax2.fill_between(x, res_T2.best_fit-dely, res_T2.best_fit+dely, color="#ABABAB")
    #plt.show()
    
    
    #params_mix = {**res_T1.best_values, **res_T2.best_values}
    params_mix = res_T1.params.copy()
    params_mix.update(res_T2.params)
    params_mix['T1center'].vary = False
    params_mix['T1sigma'].vary = False
    params_mix['T2center'].vary = False
    params_mix['T2sigma'].vary = False
    
    #params_mix['T1amplitude'].min = 10
    
    #peak1 = lmfit.models.GaussianModel(prefix='p1')
    #peak2 = lmfit.models.GaussianModel(prefix='p2')
    model = model_T1 + model_T2
    # TODO set params based on previous fitting
    res_mix = model.fit(freqs_Mix, x=x, params=params_mix)
    print(res_mix.fit_report())
    
    
    plt.sca(axM)
    res_mix.plot_fit()
    
    dely = res_mix.eval_uncertainty(sigma=3)
    axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")
    #plt.show()
    
    
    plt.figure()
    fig, (ax1r, ax2r, axMr) = plt.subplots(3, 1, sharex=True, sharey=True)
    plt.sca(ax1r)
    res_T1.plot_residuals()
    plt.sca(ax2r)
    res_T2.plot_residuals()
    plt.sca(axMr)
    res_mix.plot_residuals()
    
    print('T2/T1 =', res_mix.params['T2amplitude'].value/res_mix.params['T1amplitude'].value)
    
    T1amplitude = res_mix.params['T1amplitude'].value
    T2amplitude = res_mix.params['T2amplitude'].value
    
    print('Proportions based on Gaussians')
    print('% of Type 1:', T1amplitude/(T1amplitude+T2amplitude))
    print('% of Type 2:', T2amplitude/(T1amplitude+T2amplitude))



## Plot histograms

#-----------------------------------------------------------------------------

bin_width = 0.005
bin_edges = np.arange(0.095, 0.35+bin_width, bin_width)
bin_centers = np.arange(0.095+bin_width/2, 0.35+bin_width/2, bin_width)
#x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centres
#y = Mix

(hc1, _) = np.histogram(T1, bins=bin_edges)
(hc2, _) = np.histogram(T2, bins=bin_edges)
(hc3, _) = np.histogram(Mix, bins=bin_edges)

#-----------------------------------------------------------------------------

# compute pair-wise emds between the 3 histograms
max_emd = bin_edges[-1] - bin_edges[0]
emd_32 = sum(abs(np.cumsum(hc3/sum(hc3))-np.cumsum(hc2/sum(hc2))))*bin_width*max_emd
emd_31 = sum(abs(np.cumsum(hc3/sum(hc3))-np.cumsum(hc1/sum(hc1))))*bin_width*max_emd
emd_21 = sum(abs(np.cumsum(hc2/sum(hc2))-np.cumsum(hc1/sum(hc1))))*bin_width*max_emd

print("Proportions based on Earth Mover's Distance:")
print('% of Type 1:', 1-emd_31/emd_21)
print('% of Type 2:', 1-emd_32/emd_21)

print('Proportions based on counts')
print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))


print('--------------------------------------------------------------------------------\n\n')



# Plot proportions based on counts using stacked bars
plt.figure()
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)
#plt.subplot(2,1,1)
#br = plt.hist(np.nan_to_num([hc3*hc1/(hc1+hc2)/max(hc3), hc3*hc2/(hc1+hc2)/max(hc3)]), bin_edges)#, histtype='barstacked')
prop_T1 = np.nan_to_num(hc3*hc1/(hc1+hc2)/max(hc3))
prop_T2 = np.nan_to_num(hc3*hc2/(hc1+hc2)/max(hc3))

ax0.stackplot(bin_centers, np.vstack((hc1/(hc1+hc2), hc2/(hc1+hc2))))
ax0.set_title('Proportions of Type 1 and Type 2 vs T1GRS')


br = ax1.bar(bin_centers, prop_T1, width=bin_width, label='Type 1')
br = ax1.bar(bin_centers, prop_T2, width=bin_width, bottom=prop_T1, label='Type 2')
ax1.legend()

# plot ratios of type 2 to type 1 and 1 to 2
ax1.plot(bin_centers, hc1/(hc1+hc2), linewidth=2)
ax1.plot(bin_centers, hc2/(hc1+hc2), linewidth=2)
ax1.set_title('Proportions based on counts')
ax1.set_xlim([0.095, 0.35])
ax1.set_ylim([0, 1])

#plt.subplot(2,1,2)
# plot original histograms using counts 
#ax2.hist(T1, bin_edges)
#ax2.hist(T2, bin_edges)
#ax2.hist(Mix, bin_edges)
ax2.hist([T1, T2, Mix], bin_edges, histtype='step', linewidth='3', density=True)
ax2.set_xlim([0.095, 0.35])
ax2.set_xlabel('Tyoe 1 GRS')



## Fit KDE to populations

if False:
    # Plot all available kernels
    X_plot = np.linspace(-6, 6, 1000)[:, None]
    X_src = np.zeros((1, 1))
    
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)
    
    
    def format_func(x, loc):
        if x == 0:
            return '0'
        elif x == 1:
            return 'h'
        elif x == -1:
            return '-h'
        else:
            return '%ih' % x
    
    for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
                                'exponential', 'linear', 'cosine']):
        axi = ax.ravel()[i]
        log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
        axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
        axi.text(-2.6, 0.95, kernel)
    
        axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axi.xaxis.set_major_locator(plt.MultipleLocator(1))
        axi.yaxis.set_major_locator(plt.NullLocator())
    
        axi.set_ylim(0, 1.05)
        axi.set_xlim(-2.9, 2.9)
    
    ax[0, 1].set_title('Available Kernels')


#-----------------------------------------------------------------------------
# Plot a 1D density example

if False:
    # Specify bins
    bin_width = 0.001
    bin_edges = np.arange(0.095, 0.35+bin_width, bin_width)
    x = bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centres
    y = Mix
    (HC_mix, _) = np.histogram(Mix, bins=bin_edges)

if False:
    N = data.count()[0]
    n_bins = int(np.floor(np.sqrt(N)))
    (HC_mix, bins) = np.histogram(Mix, bins=n_bins)  # Returns bin_edges
    x = (bins[:-1] + bins[1:]) / 2  # Bin centres
    
HC_mix = freqs_Mix

#-----------------------------------------------------------------------------

bw = 0.005

#fig, ax = plt.subplots()
#ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
#        label='input distribution')
fig, axes = plt.subplots(3, 1, sharex=True) #, squeeze=False)

X_plot = np.linspace(0.1, 0.35, 1000)[:, np.newaxis]

kdes = {}
labels = ['Type 1', 'Type 2', 'Mixture']

for data, label, ax in zip([T1, T2, Mix], labels, axes):
    
    kdes[label] = {}
    X = data[:, np.newaxis]

    for kernel in ['gaussian', 'tophat', 'epanechnikov']:
        kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
                label="kernel = '{0}'; bandwidth = {1}".format(kernel, bw))
        kdes[label][kernel] = kde  #np.exp(log_dens)
    
    #ax.text(6, 0.38, "N={0} points".format(N))
    
    ax.legend(loc='upper left')
    ax.plot(X, -0.5 - 5 * np.random.random(X.shape[0]), '.')
    ax.set_ylabel(label)
    
    #ax.set_xlim(-4, 9)
    #ax.set_ylim(-0.02, 0.4)
    #plt.show()


#sp.interpolate.interp1d(X, y, X_new, y_new)

kernel = 'gaussian' #'epanechnikov'

def kde_T1(x, amp_T1):
    return amp_T1 * np.exp(kdes['Type 1'][kernel].score_samples(x[:, np.newaxis]))

def kde_T2(x, amp_T2):
    return amp_T2 * np.exp(kdes['Type 2'][kernel].score_samples(x[:, np.newaxis]))


from lmfit import Model

model_T1 = Model(kde_T1)
model_T2 = Model(kde_T2)


plt.figure()
fig, (axP, axM, axR, axI) = plt.subplots(4, 1, sharex=True, sharey=False)

model = model_T1 + model_T2
params_mix = model.make_params()
params_mix['amp_T1'].value = 1
params_mix['amp_T2'].value = 1

res_mix = model.fit(HC_mix, x=x, params=params_mix)

plt.sca(axM)
res_mix.plot_fit()

dely = res_mix.eval_uncertainty(sigma=3)
axM.fill_between(x, res_mix.best_fit-dely, res_mix.best_fit+dely, color="#ABABAB")
#plt.show()

plt.sca(axR)
res_mix.plot_residuals()


amp_T1 = res_mix.params['amp_T1'].value
amp_T2 = res_mix.params['amp_T2'].value
kde1 = kde_T1(x, amp_T1)
kde2 = kde_T2(x, amp_T2)
axP.stackplot(x, np.vstack((kde1/(kde1+kde2), kde2/(kde1+kde2))), labels=labels[:-1])
legend = axP.legend(facecolor='grey')
#legend.get_frame().set_facecolor('grey')
axP.set_title('Proportions of Type 1 and Type 2 vs T1GRS')

#plt.sca(axI)
axI.plot(x, kde1, label='Type 1')
axI.plot(x, kde2, label='Type 2')

print(res_mix.fit_report())
print('T2/T1 =', res_mix.params['amp_T2'].value/res_mix.params['amp_T1'].value)
print('')


print("Proportions based on Earth Mover's Distance:")
print('% of Type 1:', 1-emd_31/emd_21)
print('% of Type 2:', 1-emd_32/emd_21)

print('Proportions based on counts')
print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))

print('Proportions based on KDEs')
print('% of Type 2:', amp_T1/(amp_T1+amp_T2))
print('% of Type 2:', amp_T2/(amp_T1+amp_T2))

print('\nParameter confidence intervals:')
print(res_mix.ci_report())  # --> res_mix.ci_out # See also res_mix.conf_interval()

print('--------------------------------------------------------------------------------\n\n')



if True:
    # Bootstrap 18:51?
    # Interpolate the cdfs at the same points for comparison
    x_T1 = [0.095, *sorted(T1), 0.35]
    y_T1 = np.linspace(0, 1, len(x_T1))
    (iv, ii) = np.unique(x_T1, return_index=True)
    i_cdf1 = np.interp(bin_centers, iv, y_T1[ii])
    
    x_T2 = [0.095, *sorted(T2), 0.35]
    y_T2 = np.linspace(0, 1, len(x_T2))
    (iv, ii) = np.unique(x_T2, return_index=True)
    i_cdf2 = np.interp(bin_centers, iv, y_T2[ii])
    
    x_Mix = [0.095, *sorted(Mix), 0.35]
    y_Mix = np.linspace(0, 1, len(x_Mix))
    (iv, ii) = np.unique(x_Mix, return_index=True)
    i_cdf3 = np.interp(bin_centers, iv, y_Mix[ii])
    
    print('Root mean square fit of CDF with EMDs:')
    p_emd_1 = 1-emd_31/emd_21  # Proportions of Type 1 in Mix based on EMD
    p_emd_2 = 1-emd_32/emd_21  # Proportions of Type 2 in Mix based on EMD
    print(np.sqrt(sum((i_cdf3-(p_emd_2*i_cdf2 + p_emd_1*i_cdf1))**2))/len(i_cdf3))
    print('Root mean square fit of CDF with counts:')
    print(np.sqrt(sum((i_cdf3-(0.6085*i_cdf2+0.3907*i_cdf1))**2))/len(i_cdf3))
    
    ## Some test of the quality of the fit using CDFs
    # plot experimental cdfs
    plt.figure()
    plt.plot(x_T1, y_T1, label='Type 1')
    plt.plot(x_T2, y_T2, label='Type 2')
    plt.plot(x_Mix, y_Mix, label='Mixture')
    
    # plot interpolated CDFs
    plt.plot(bin_centers, i_cdf1, '.-', label='Type 1 CDF')
    plt.plot(bin_centers, i_cdf2, '.-', label='Type 2 CDF')
    plt.plot(bin_centers, i_cdf3, '.-', label='Mixture CDF')
    
    # plot linear combinations of the two original distributions
    plt.plot(bin_centers, p_emd_2*i_cdf2 + p_emd_1*i_cdf1, 'p-', label='EMD-based combination')  # emds
    plt.plot(bin_centers, 0.6085*i_cdf2 + 0.3907*i_cdf1, 'o-', label='HC-based combination')  # counts
    #axis([0.095 0.35 -0.01 1.01])
    plt.legend()
    
    ## addtional test normalised bar plot using the proportion from the EMD 
    plt.figure()
    plt.subplot(2,1,1)
    # br=bar(bin_centers,[hc1*0.2429; hc2*0.7563]'./max(sum([hc1*0.2429; hc2*0.7563])),'stacked');
    #br = plt.bar(bin_centers, np.vstack((hc1*p_emd_1, hc2*p_emd_2))/max(np.sum(np.vstack((hc1*p_emd_1, hc2*p_emd_2)), axis=0)), 'stacked')
    norm_fact = np.sum(np.vstack((hc1*p_emd_1, hc2*p_emd_2)), axis=0)
    plt.bar(bin_centers, height=hc1*p_emd_1/norm_fact, bottom=hc2*p_emd_2/norm_fact, width=bin_width)
    plt.bar(bin_centers, height=hc2*p_emd_2/norm_fact, width=bin_width)
    
    
    plt.subplot(2,1,2)
    plt.bar(bin_centers, height=hc3/max(hc3), width=bin_width)  # bar(bin_centers,hc3/max(hc3))


