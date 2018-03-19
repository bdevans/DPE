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

# Configure plots
PLOT_RELATIVE_ERROR = True
PLOT_ABSOLUTE_ERROR = True
PLOT_STANDARD_DEVIATION = True
ADJUST_EXCESS = True

LABELS = ['Means', 'Excess', 'EMD', 'KDE']
COLOURS = ['r', 'g', 'b', 'k']
average = np.mean
deviation = np.std

PROPORTIONS_T1D = np.load('data/proportions.npy')
SAMPLE_SIZES = np.load('data/sample_sizes.npy')
PROPORTIONS_T2D = PROPORTIONS_T1D[::-1]

MEANS_T1D = np.load('data/means.npy')
MEANS_T2D = 1 - MEANS_T1D
EXCESS_T1D = np.load('data/excess.npy')
EXCESS_T2D = 1 - EXCESS_T1D
EMD_31 = np.load('data/emd_31.npy')
EMD_32 = np.load('data/emd_32.npy')
KDE_T1D = np.load('data/kde.npy')
KDE_T2D = 1 - KDE_T1D

if ADJUST_EXCESS:
    EXCESS_T1D /= 0.92  # adjusted for fact underestimates by 8%


if PLOT_RELATIVE_ERROR:
    # Relative error

    LEVELS = np.array([5.0])  # Percentage relative error

    plt.figure()
    plt.title('Maximum absolute relative average percentage error\nContours at {}'.format(LEVELS))
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')

    relative_error_EMD_T1 = 100*(average(1-EMD_31, axis=2)-PROPORTIONS_T1D)/PROPORTIONS_T1D
    relative_error_EMD_T2 = 100*(average(1-EMD_32, axis=2)-PROPORTIONS_T2D)/PROPORTIONS_T2D
    max_relative_error_EMD = np.maximum(np.abs(relative_error_EMD_T1), np.abs(relative_error_EMD_T2))

    #if run_means:
    relative_error_MEANS_T1 = 100*(average(MEANS_T1D, axis=2)-PROPORTIONS_T1D)/PROPORTIONS_T1D
    relative_error_MEANS_T2 = 100*(average(MEANS_T2D, axis=2)-PROPORTIONS_T2D)/PROPORTIONS_T2D
    max_relative_error_MEANS = np.maximum(np.abs(relative_error_MEANS_T1), np.abs(relative_error_MEANS_T2))

    #if run_excess:
    relative_error_EXCESS_T1 = 100*(average(EXCESS_T1D, axis=2)-PROPORTIONS_T1D)/PROPORTIONS_T1D
    relative_error_EXCESS_T2 = 100*(average(EXCESS_T2D, axis=2)-PROPORTIONS_T2D)/PROPORTIONS_T2D
    max_relative_error_EXCESS = np.maximum(np.abs(relative_error_EXCESS_T1), np.abs(relative_error_EXCESS_T2))

    relative_error_KDE_T1 = 100*(average(KDE_T1D, axis=2)-PROPORTIONS_T1D)/PROPORTIONS_T1D
    relative_error_KDE_T2 = 100*(average(KDE_T2D, axis=2)-PROPORTIONS_T2D)/PROPORTIONS_T2D
    max_relative_error_KDE = np.maximum(np.abs(relative_error_KDE_T1), np.abs(relative_error_KDE_T2))


    datasets = [max_relative_error_MEANS, max_relative_error_EXCESS,
                max_relative_error_EMD, max_relative_error_KDE]

    for label, colour, data in zip(LABELS, COLOURS, datasets):
        CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        CS.collections[0].set_label(label)
    plt.legend()


    # Plot shaded regions for each method on individual subplots
    #SHADING_LEVELS = np.array([0, 2.5, 5.0, 10, 15, 25, 50, 100, 150])
    SHADING_LEVELS = np.arange(0, 100, 2.5)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.suptitle('Maximum absolute relative average percentage error\nContours at {}'.format(LEVELS))
    for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
        ax.set_title(label)
        CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data, SHADING_LEVELS,
                         cmap='viridis_r', extend='max')
        ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        fig.colorbar(CS, ax=ax)


if PLOT_ABSOLUTE_ERROR:
    # Absolute error

    LEVELS = np.array([0.02])  # Percentage relative error

    plt.figure()
    plt.title('Maximum absolute average proportion error\nContours at {}'.format(LEVELS))
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')

    #if run_EMD:
    abs_error_EMD_T1 = np.abs(average(1-EMD_31, axis=2)-PROPORTIONS_T1D)
    abs_error_EMD_T2 = np.abs(average(1-EMD_32, axis=2)-PROPORTIONS_T2D)
    max_abs_error_EMD = np.maximum(abs_error_EMD_T1, abs_error_EMD_T2)

    #if run_means:
    abs_error_MEANS_T1 = np.abs(average(MEANS_T1D, axis=2)-PROPORTIONS_T1D)
    abs_error_MEANS_T2 = np.abs(average(MEANS_T2D, axis=2)-PROPORTIONS_T2D)
    max_abs_error_MEANS = np.maximum(abs_error_MEANS_T1, abs_error_MEANS_T2)

    #if run_excess:
    abs_error_EXCESS_T1 = np.abs(average(EXCESS_T1D, axis=2)-PROPORTIONS_T1D)
    abs_error_EXCESS_T2 = np.abs(average(EXCESS_T2D, axis=2)-PROPORTIONS_T2D)
    max_abs_error_EXCESS = np.maximum(abs_error_EXCESS_T1, abs_error_EXCESS_T2)

    abs_error_KDE_T1 = np.abs(average(KDE_T1D, axis=2)-PROPORTIONS_T1D)
    abs_error_KDE_T2 = np.abs(average(KDE_T2D, axis=2)-PROPORTIONS_T2D)
    max_abs_error_KDE = np.maximum(abs_error_KDE_T1, abs_error_KDE_T2)


    datasets = [max_abs_error_MEANS, max_abs_error_EXCESS,
                max_abs_error_EMD, max_abs_error_KDE]

    for label, colour, data in zip(LABELS, COLOURS, datasets):
        CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        CS.collections[0].set_label(label)
    plt.legend()

    # Plot shaded regions for each method on individual subplots
    #SHADING_LEVELS = np.array([0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1])
    SHADING_LEVELS = np.arange(0, 0.1, 0.005)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.suptitle('Maximum absolute average proportion error\nContours at {}'.format(LEVELS))
    for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
        ax.set_title(label)
        CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data, SHADING_LEVELS,
                         cmap='viridis_r', extend='max')
        ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        fig.colorbar(CS, ax=ax)


if PLOT_STANDARD_DEVIATION:

    LEVELS = np.array([0.01])

    plt.figure()
    plt.title('Maximum standard deviation\nContours at {}'.format(LEVELS))
    plt.xlabel('Proportion T1D')
    plt.ylabel('Sample size')

    #if run_means:
#    dev_means_T1 = deviation(MEANS_T1D, axis=2)
#    dev_means_T2 = deviation(MEANS_T2D, axis=2)
#    max_dev_MEANS = np.maximum(dev_means_T1, dev_means_T2)
    dev_MEANS = deviation(MEANS_T1D, axis=2)

    #if run_excess:
#    dev_excess_T1 = deviation(EXCESS_T1D, axis=2)
#    dev_excess_T2 = deviation(EXCESS_T2D, axis=2)
#    max_dev_EXCESS = np.maximum(dev_excess_T1, dev_excess_T2)
    dev_EXCESS = deviation(EXCESS_T1D, axis=2)

    #if run_EMD:
    dev_EMD_T1 = deviation(EMD_31, axis=2)
    dev_EMD_T2 = deviation(EMD_32, axis=2)
    max_dev_EMD = np.maximum(dev_EMD_T1, dev_EMD_T2)
    #dev_EMD = deviation(EMD_31, axis=2)

#    dev_kde_T1 = deviation(KDE_T1D, axis=2)
#    dev_kde_T2 = deviation(KDE_T2D, axis=2)
#    max_dev_KDE = np.maximum(dev_kde_T1, dev_kde_T2)
    dev_KDE = deviation(KDE_T1D, axis=2)


    datasets = [dev_MEANS, dev_EXCESS, max_dev_EMD, dev_KDE]

    for label, colour, data in zip(LABELS, COLOURS, datasets):
        CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        CS.collections[0].set_label(label)
    plt.legend(loc='upper left')


    # Plot shaded regions for each method on individual subplots
    #SHADING_LEVELS = np.array([0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1])#, 0.25, 0.5])
    SHADING_LEVELS = np.arange(0.005, 0.1, 0.005)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    plt.suptitle('Maximum standard deviation\nContours at {}'.format(LEVELS))
    for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
        ax.set_title(label)
        CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data, SHADING_LEVELS,
                         cmap='viridis_r', extend='both')
        ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
        fig.colorbar(CS, ax=ax)
