#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:07:52 2018

@author: ben
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.style.use('seaborn')
mpl.rc('figure', figsize=(12, 10))

proportions = np.load('data/proportions.npy')
sample_sizes = np.load('data/sample_sizes.npy')
proportions_rev = proportions[::-1]


plot_relative_error = True
plot_absolute_error = True
plot_standard_deviation = True
adjust_excess = True

norm_mat_EMD_31 = np.load('data/emd_31.npy')
norm_mat_EMD_32 = np.load('data/emd_32.npy')
means_T1D = np.load('data/means.npy')
excess_T1D = np.load('data/excess.npy')
kde_T1D = np.load('data/kde.npy')

if adjust_excess:
    excess_T1D = excess_T1D/0.92  # adjusted for fact underestimates by 8%


if plot_relative_error:
    # Relative error
    plt.figure()
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')
    #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
    #plt.colorbar()
    levels = np.array([5.0])  # np.array([0.1, 1.0])  # Percentage relative error

    relative_error_EMD_T1 = 100*(np.median(1-norm_mat_EMD_31, axis=2)-proportions)/proportions
    relative_error_EMD_T2 = 100*(np.median(1-norm_mat_EMD_32, axis=2)-proportions_rev)/proportions_rev
    max_relative_error_EMD = np.maximum(np.abs(relative_error_EMD_T1), np.abs(relative_error_EMD_T2))
    CS = plt.contour(proportions, sample_sizes, max_relative_error_EMD,
                     levels, colors='r')

    #if run_means:
    #relative_error_means = 100*(np.median(means_T1D/100, axis=2)-proportions)/proportions
    relative_error_means_T1 = 100*(np.median(means_T1D/100, axis=2)-proportions)/proportions
    relative_error_means_T2 = 100*(np.median(1-means_T1D/100, axis=2)-proportions_rev)/proportions_rev
    max_relative_error_means = np.maximum(np.abs(relative_error_means_T1), np.abs(relative_error_means_T2))
    CS = plt.contour(proportions, sample_sizes, max_relative_error_means,
                     levels, colors='k')

    #if run_excess:
    relative_error_excess_T1 = 100*(np.median(excess_T1D/100, axis=2)-proportions)/proportions
    relative_error_excess_T2 = 100*(np.median(1-excess_T1D/100, axis=2)-proportions_rev)/proportions_rev
    max_relative_error_excess = np.maximum(np.abs(relative_error_excess_T1), np.abs(relative_error_excess_T2))
    CS = plt.contour(proportions, sample_sizes, max_relative_error_excess,
                     levels, colors='g')

    relative_error_kde_T1 = 100*(np.median(kde_T1D, axis=2)-proportions)/proportions
    relative_error_kde_T2 = 100*(np.median(1-kde_T1D, axis=2)-proportions_rev)/proportions_rev
    max_relative_error_kde = np.maximum(np.abs(relative_error_kde_T1), np.abs(relative_error_kde_T2))
    CS = plt.contour(proportions, sample_sizes, max_relative_error_kde,
                     levels, colors='b')

    plt.contourf(proportions, sample_sizes, max_relative_error_kde, cmap='viridis_r')
    plt.colorbar()

if plot_absolute_error:
    # Absolute error
    plt.figure()
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')
    #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
    #plt.colorbar()

    levels = np.array([(0.05)]) #np.array([0.1, 1.0])  # # Percentage relative error

    #if run_EMD:
    relative_error_EMD_T1D = np.abs(((np.mean(1-norm_mat_EMD_31, axis=2))/proportions)-1)
    relative_error_EMD_T2D = np.abs(((1-(np.mean(1-norm_mat_EMD_31, axis=2)))/proportions_rev)-1)
    max_relative_error_emd_abs = np.maximum(relative_error_EMD_T1D, relative_error_EMD_T2D)
    CS = plt.contour(proportions, sample_sizes, max_relative_error_emd_abs,
                     levels, colors='r')

    #if run_means:
    relative_error_means_T1D = np.abs(((np.mean(means_T1D/100, axis=2))/proportions)-1)
    relative_error_means_T2D = np.abs(((1-(np.mean(means_T1D/100, axis=2)))/proportions_rev)-1)
    max_relative_error_means_abs = np.maximum(relative_error_means_T1D, relative_error_means_T2D)
    CS = plt.contour(proportions, sample_sizes, max_relative_error_means_abs,
                     levels, colors='k')

    #if run_excess:
    relative_error_excess_T1D = np.abs(((np.mean(excess_T1D/100, axis=2))/proportions)-1)
    relative_error_excess_T2D = np.abs(((1-(np.mean(excess_T1D/100, axis=2)))/proportions_rev)-1)
    max_relative_error_excess_abs = np.maximum(relative_error_excess_T1D, relative_error_excess_T2D)
    CS = plt.contour(proportions, sample_sizes, max_relative_error_excess_abs,
                     levels, colors='g')

    relative_error_kde_T1D = np.abs(((np.mean(kde_T1D, axis=2))/proportions)-1)
    relative_error_kde_T2D = np.abs(((1-(np.mean(kde_T1D, axis=2)))/proportions_rev)-1)
    max_relative_error_kde_abs = np.maximum(relative_error_kde_T1D, relative_error_kde_T2D)
    CS = plt.contour(proportions, sample_sizes, max_relative_error_kde_abs,
                     levels, colors='b')

    plt.contourf(proportions, sample_sizes, max_relative_error_kde_abs, cmap='viridis_r')
    plt.colorbar()

if plot_standard_deviation:
    plt.figure()
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')
    #plt.contourf(proportions, sample_sizes, median_error, cmap='viridis_r')
    #plt.colorbar()

    levels = np.array([3.0]) #np.array([0.1, 1.0])  # # Percentage relative error

    #if run_EMD:
    adjusted_EMD = 100*(1-norm_mat_EMD_31)
    dev_EMD_t1 = np.std(adjusted_EMD, axis=2)
    dev_EMD_t2 = np.std(100-adjusted_EMD, axis=2)
    max_dev_EMD = np.abs(np.maximum(dev_EMD_t1, dev_EMD_t2))
    CS = plt.contour(proportions, sample_sizes, max_dev_EMD, levels, colors='r')

    #if run_means:
    dev_means_t1 = np.std(means_T1D, axis=2)
    dev_means_t2 = np.std(100-means_T1D, axis=2)
    max_dev_means = np.abs(np.maximum(dev_means_t1, dev_means_t2))
    CS = plt.contour(proportions, sample_sizes, max_dev_means, levels, colors='k')

    #if run_excess:
    dev_excess_t1 = np.std(excess_T1D, axis=2)
    dev_excess_t2 = np.std(100-excess_T1D, axis=2)
    max_dev_excess = np.abs(np.maximum(dev_excess_t1, dev_excess_t2))
    CS = plt.contour(proportions, sample_sizes, max_dev_excess, levels, colors='g')

    dev_kde_t1 = np.std(kde_T1D, axis=2)
    dev_kde_t2 = np.std(100-kde_T1D, axis=2)
    max_dev_kde = np.abs(np.maximum(dev_kde_t1, dev_kde_t2))
    CS = plt.contour(proportions, sample_sizes, max_dev_kde, levels, colors='b')

    plt.contourf(proportions, sample_sizes, max_dev_kde, cmap='viridis_r')
    plt.colorbar()