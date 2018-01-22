#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

@author: ben
"""

import time
import math

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

# Arrays of T1GRS scores for each group
T1 = data.loc[data['type'] == 1, 'T1GRS'].as_matrix()
T2 = data.loc[data['type'] == 2, 'T1GRS'].as_matrix()
Mix = data.loc[data['type'] == 3, 'T1GRS'].as_matrix()
scores = {'T1': T1, 'T2': T2, 'Mix': Mix}  # Raw T1GRS scores

#------------------------------ Bin the data ---------------------------------
N = data.count()[0]
bin_width = 0.005
bin_edges = np.arange(0.095, 0.35+bin_width, bin_width)
#bin_centers = np.arange(0.095+bin_width/2, 0.35+bin_width/2, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centres

(hc1, _) = np.histogram(T1, bins=bin_edges)
(hc2, _) = np.histogram(T2, bins=bin_edges)
(hc3, _) = np.histogram(Mix, bins=bin_edges)
counts = {'T1': hc1, 'T2': hc2, 'Mix': hc3}  # Binned score frequencies




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

# emds computed with histograms (compute pair-wise emds between the 3 histograms)
max_emd = bin_edges[-1] - bin_edges[0]
emd_21 = sum(abs(np.cumsum(hc2/sum(hc2))-np.cumsum(hc1/sum(hc1))))*bin_width*max_emd
emd_31 = sum(abs(np.cumsum(hc3/sum(hc3))-np.cumsum(hc1/sum(hc1))))*bin_width*max_emd
emd_32 = sum(abs(np.cumsum(hc3/sum(hc3))-np.cumsum(hc2/sum(hc2))))*bin_width*max_emd

# emds computed with interpolated cdfs
iemd_21=sum(abs(i_cdf2-i_cdf1))*bin_width*max_emd
iemd_31=sum(abs(i_cdf3-i_cdf1))*bin_width*max_emd
iemd_32=sum(abs(i_cdf3-i_cdf2))*bin_width*max_emd

print("Proportions based on Earth Mover's Distance:")
print('% of Type 1:', 1-emd_31/emd_21)
print('% of Type 2:', 1-emd_32/emd_21)

print('Proportions based on counts')
print('% of Type 1:', np.nansum(hc3*hc1/(hc1+hc2))/sum(hc3))
print('% of Type 2:', np.nansum(hc3*hc2/(hc1+hc2))/sum(hc3))


print('--------------------------------------------------------------------------------\n\n')


max_rn = 100
sample_sizes = np.array(range(100, 3100, 100))
proportions = np.arange(0.01, 1.01, 0.02)

emd_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), max_rn))
rms_dev_from_fit = np.zeros((len(sample_sizes), len(proportions), max_rn))
mat_emd_31 = np.zeros((len(sample_sizes), len(proportions), max_rn))
mat_emd_32 = np.zeros((len(sample_sizes), len(proportions), max_rn))

t = time.time()  # Start timer
for rn in range(max_rn):
    for s, sample_size in enumerate(sample_sizes):
        for p, prop_T1 in enumerate(proportions):
            
            nT1 = int(round(sample_size * prop_T1))
            nT2 = sample_size - nT1
            
            # Random sample from T1
            R1 = np.random.choice(T1, nT1, replace=True)
            
            # Random sample from T2
            R2 = np.random.choice(T2, nT2, replace=True)

            # Bootstrap mixture
            RM = np.sort(np.concatenate((R1, R2)))
            xRM = np.linspace(0, 1, num=len(RM), endpoint=True)
            
            # Interpolated cdf (to compute emd)
            x = [0.095, *RM, 0.35]
            y = np.linspace(0, 1, num=len(x), endpoint=True)
            (iv, ii) = np.unique(x, return_index=True)
            si_cdf3 = np.interp(bin_centers, iv, y[ii])
            
            # Compute EMDs
            iemd_31 = sum(abs(si_cdf3-i_cdf1)) * bin_width * max_emd;
            iemd_32 = sum(abs(si_cdf3-i_cdf2)) * bin_width * max_emd;
            mat_emd_31[s, p, rn] = iemd_31  # emds to compute proportions
            mat_emd_32[s, p, rn] = iemd_32  # emds to compute proportions
            
            EMD_diff = si_cdf3 - ((1-iemd_31/iemd_21)*i_cdf1 + (1-iemd_32/iemd_21)*i_cdf2)
            emd_dev_from_fit[s, p, rn] = sum(EMD_diff) * bin_width * max_emd  # deviations from fit measured with emd
            rms_dev_from_fit[s, p, rn] = math.sqrt(sum(EMD_diff**2)) / len(si_cdf3)  # deviations from fit measured with rms
elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds'.format(elapsed))

# Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
norm_mat_EMD_31 = mat_emd_31 / emd_21
norm_mat_EMD_32 = mat_emd_32 / emd_21
norm_EMD_dev = emd_dev_from_fit / emd_21

# Deviation from fit
median_error = np.median(norm_EMD_dev, axis=2)
plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
plt.colorbar()

levels = 100 * np.array([0.001, 0.01]) * np.amax(norm_EMD_dev)
CS = plt.contour(proportions, sample_sizes, 100*median_error, levels)
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('Proportion (Type 1)')
plt.ylabel('Sample size')
plt.title('Median propotion error from true proportion (as a % of maximum EMD error)\nContours represent maximum error')


# Error p2
plt.figure()
rel_err_31 = 100*(np.median((1-norm_mat_EMD_31), axis=2) - proportions)/proportions
plt.contourf(proportions, sample_sizes, rel_err_31, cmap='viridis_r')
plt.colorbar()
#input_prop=repmat(0.01:0.02:0.99,30,1,10); %set proportion
#surf(100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3))
#hold on

# 1 & 5% relative error contour the other proportion
CS = plt.contour(proportions, sample_sizes, rel_err_31, [1, 5])
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('Proportion (Type 1)')
plt.ylabel('Sample size')
plt.title('Relative % error from Type 1 population')
#contour3(100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3),[5 5],'r','LineWidth',3)
#hold off
#set(gca,'xtick',linspace(1,50,21),'xticklabel',[1 5:5:99 99])
#xlabel('% proportion')
#set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
#ylim([1 30])
#ylabel('sample size')



# Error p1
plt.figure()
proportions_rev = proportions[::-1]
rel_err_32 = 100*(np.median((1-norm_mat_EMD_32), axis=2) - proportions_rev)/proportions_rev
plt.contourf(proportions, sample_sizes, rel_err_32, cmap='viridis_r')
plt.colorbar()
#input_prop=repmat(0.99:-0.02:0.01,30,1,10); %set original proportion
#surf(100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3))
#hold on

# 1 & 5% relative error contour the other proportion
CS = plt.contour(proportions, sample_sizes, rel_err_32, [1, 5])
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('Proportion (Type 1)')
plt.ylabel('Sample size')
plt.title('Relative % error from Type 2 population')
#contour3(100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3),[5 5],'r','LineWidth',3)
#hold off
#set(gca,'xtick',linspace(1,50,21),'xticklabel',[99 99:-5:5 1])
#xlabel('% proportion')
#set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
#ylim([1 30])
#ylabel('sample size')



# Max Error
#ers = np.zeros((len(sample_sizes), len(proportions), 2))
#tmp = np.expand_dims(proportions, axis=1)
#props = np.tile(tmp, (len(sample_sizes), 1, max_rn))
#ers[:,:,0] = 100*np.median(((1-mat_emd_32/iemd_21)-props)/props, axis=2)
#
#ers = np.zeros((len(sample_sizes), len(proportions), 2))
#tmp = np.expand_dims(proportions, axis=0)
#props = np.tile(tmp, (len(sample_sizes), 1))
#ers[:,:,0] = 100*(np.median((1-mat_emd_32/iemd_21), axis=2)-props)/props

ers = np.zeros((len(sample_sizes), len(proportions), 2))
# N.B. Swapped indicies
ers[:,:,0] = 100*(np.median(1-mat_emd_31/iemd_21, axis=2) - proportions_rev)/proportions_rev
ers[:,:,1] = 100*(np.median(1-mat_emd_32/iemd_21, axis=2) - proportions)/proportions
max_ers = np.amax(ers, axis=2)

plt.figure()
plt.contourf(proportions, sample_sizes, max_ers, cmap='viridis_r')
plt.colorbar()

#input_prop=repmat(0.99:-0.02:0.01,30,1,10); %set original proportion
#ers(:,:,1)=100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3);
#input_prop=repmat(0.01:0.02:0.99,30,1,10); %set proportion
#ers(:,:,2)=100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3);
#surf(max(ers,[],3))
#hold on

# 1 & 5% relative error contour the max error proportion
CS = plt.contour(proportions, sample_sizes, max_ers, [1, 5])
plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('Proportion (Type 1)')
plt.ylabel('Sample size')
plt.title('Maximum % relative error from either population')

#contour3(max(ers,[],3),[5 5],'r','LineWidth',3)
#hold off
#set(gca,'xtick',linspace(1,50,21),'xticklabel',[1 5:5:99 99])
#xlabel('% proportion')
#set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
#ylim([1 30])
#ylabel('sample size')







if False:
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
    
