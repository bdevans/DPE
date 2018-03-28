#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:07:52 2018

@author: ben
"""
import os.path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_bootstrap_results(metric):

    PROPORTIONS_T1D = np.load('results/proportions_{}.npy'.format(metric))
    SAMPLE_SIZES = np.load('results/sample_sizes_{}.npy'.format(metric))
    PROPORTIONS_T2D = PROPORTIONS_T1D[::-1]

    if os.path.isfile('results/means_{}.npy'.format(metric)):
        MEANS_T1D = np.load('results/means_{}.npy'.format(metric))
        MEANS_T2D = 1 - MEANS_T1D
        PLOT_MEANS = True
    if os.path.isfile('results/excess_{}.npy'.format(metric)):
        EXCESS_T1D = np.load('results/excess_{}.npy'.format(metric))
        EXCESS_T2D = 1 - EXCESS_T1D
        if ADJUST_EXCESS:
            EXCESS_T1D /= 0.92  # adjusted for fact underestimates by 8%
        PLOT_EXCESS = True
    if os.path.isfile('results/emd_31_{}.npy'.format(metric)) and \
       os.path.isfile('results/emd_32_{}.npy'.format(metric)):
        EMD_31 = np.load('results/emd_31_{}.npy'.format(metric))
        EMD_32 = np.load('results/emd_32_{}.npy'.format(metric))
        PLOT_EMD = True
    if os.path.isfile('results/kde_{}.npy'.format(metric)):
        KDE_T1D = np.load('results/kde_{}.npy'.format(metric))
        KDE_T2D = 1 - KDE_T1D
        PLOT_KDE = True


    error_MEANS_T1 = average(MEANS_T1D, axis=2) - PROPORTIONS_T1D
    error_MEANS_T2 = average(MEANS_T2D, axis=2) - PROPORTIONS_T2D
    error_EXCESS_T1 = average(EXCESS_T1D, axis=2) - PROPORTIONS_T1D
    error_EXCESS_T2 = average(EXCESS_T2D, axis=2) - PROPORTIONS_T2D
    error_EMD_T1 = average(1-EMD_31, axis=2) - PROPORTIONS_T1D
    error_EMD_T2 = average(1-EMD_32, axis=2) - PROPORTIONS_T2D
    error_KDE_T1 = average(KDE_T1D, axis=2) - PROPORTIONS_T1D
    error_KDE_T2 = average(KDE_T2D, axis=2) - PROPORTIONS_T2D

    if PLOT_RELATIVE_ERROR:

        LEVELS = np.array([5.0])  # Percentage relative error

        relative_error_MEANS_T1 = 100*error_MEANS_T1/PROPORTIONS_T1D
        relative_error_MEANS_T2 = 100*error_MEANS_T2/PROPORTIONS_T2D
        max_relative_error_MEANS = np.maximum(np.abs(relative_error_MEANS_T1),
                                              np.abs(relative_error_MEANS_T2))

        relative_error_EXCESS_T1 = 100*error_EXCESS_T1/PROPORTIONS_T1D
        relative_error_EXCESS_T2 = 100*error_EXCESS_T2/PROPORTIONS_T2D
        max_relative_error_EXCESS = np.maximum(np.abs(relative_error_EXCESS_T1),
                                               np.abs(relative_error_EXCESS_T2))

        relative_error_EMD_T1 = 100*error_EMD_T1/PROPORTIONS_T1D
        relative_error_EMD_T2 = 100*error_EMD_T2/PROPORTIONS_T2D
        max_relative_error_EMD = np.maximum(np.abs(relative_error_EMD_T1),
                                            np.abs(relative_error_EMD_T2))

        relative_error_KDE_T1 = 100*error_KDE_T1/PROPORTIONS_T1D
        relative_error_KDE_T2 = 100*error_KDE_T2/PROPORTIONS_T2D
        max_relative_error_KDE = np.maximum(np.abs(relative_error_KDE_T1),
                                            np.abs(relative_error_KDE_T2))

        datasets = [max_relative_error_MEANS, max_relative_error_EXCESS,
                    max_relative_error_EMD, max_relative_error_KDE]

        # Plot contours for each method on one figure
        plt.figure()
        plt.title('{} : Maximum relative error (%)\nContours at {}'.format(metric, LEVELS))
        plt.xlabel('Proportion T1D')
        plt.ylabel('Sample size')

        for label, colour, data in zip(LABELS, COLOURS, datasets):
            CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            CS.collections[0].set_label(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/relative_{}.png'.format(metric))

        # Plot shaded regions for each method on individual subplots
        SHADING_LEVELS = np.arange(0, 100, 5)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        plt.suptitle('{} : Maximum relative error (%) [Contours at {}]'.format(metric, LEVELS))
        for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
            ax.set_title(label)
            CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                             SHADING_LEVELS, cmap='viridis_r', extend='max')
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            fig.colorbar(CS, ax=ax)
        plt.tight_layout()
        plt.savefig('figs/relative_sub_{}.png'.format(metric))


    if PLOT_ABSOLUTE_ERROR:

        LEVELS = np.array([0.02])  # Percentage relative error

        max_abs_error_MEANS = np.maximum(np.abs(error_MEANS_T1),
                                         np.abs(error_MEANS_T2))

        max_abs_error_EXCESS = np.maximum(np.abs(error_EXCESS_T1),
                                          np.abs(error_EXCESS_T2))

        max_abs_error_EMD = np.maximum(np.abs(error_EMD_T1), np.abs(error_EMD_T2))

        max_abs_error_KDE = np.maximum(np.abs(error_KDE_T1), np.abs(error_KDE_T2))

        datasets = [max_abs_error_MEANS, max_abs_error_EXCESS,
                    max_abs_error_EMD, max_abs_error_KDE]

        # Plot contours for each method on one figure
        plt.figure()
        plt.title('{} : Maximum absolute error\nContours at {}'.format(metric, LEVELS))
        plt.xlabel('Proportion T1D')
        plt.ylabel('Sample size')

        for label, colour, data in zip(LABELS, COLOURS, datasets):
            CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            CS.collections[0].set_label(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig('figs/absolute_{}.png'.format(metric))

        # Plot shaded regions for each method on individual subplots
        SHADING_LEVELS = np.arange(0, 0.1, 0.005)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        plt.suptitle('{} : Maximum absolute error [Contours at {}]'.format(metric, LEVELS))
        for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
            ax.set_title(label)
            CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                             SHADING_LEVELS, cmap='viridis_r', extend='max')
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            fig.colorbar(CS, ax=ax)
        plt.tight_layout()
        plt.savefig('figs/absolute_sub_{}.png'.format(metric))


    if PLOT_STANDARD_DEVIATION:

        LEVELS = np.array([0.02])

        dev_MEANS = deviation(MEANS_T1D, axis=2)

        dev_EXCESS = deviation(EXCESS_T1D, axis=2)

        dev_EMD_T1 = deviation(EMD_31, axis=2)
        dev_EMD_T2 = deviation(EMD_32, axis=2)
        max_dev_EMD = np.maximum(dev_EMD_T1, dev_EMD_T2)

        dev_KDE = deviation(KDE_T1D, axis=2)

        datasets = [dev_MEANS, dev_EXCESS, max_dev_EMD, dev_KDE]

        # Plot shaded regions for each method on individual subplots
        plt.figure()
        plt.title('{} : Maximum standard deviation\nContours at {}'.format(metric, LEVELS))
        plt.xlabel('Proportion T1D')
        plt.ylabel('Sample size')

        for label, colour, data in zip(LABELS, COLOURS, datasets):
            CS = plt.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            CS.collections[0].set_label(label)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('figs/deviation_{}.png'.format(metric))

        # Plot shaded regions for each method on individual subplots
        SHADING_LEVELS = np.arange(0.005, 0.1, 0.005)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        plt.suptitle('{} : Maximum standard deviation [Contours at {}]'.format(metric, LEVELS))
        for ax, label, colour, data in zip(axes.ravel(), LABELS, COLOURS, datasets):
            ax.set_title(label)
            CS = ax.contourf(PROPORTIONS_T1D, SAMPLE_SIZES, data,
                             SHADING_LEVELS, cmap='viridis_r', extend='both')
            ax.contour(PROPORTIONS_T1D, SAMPLE_SIZES, data, LEVELS, colors=colour)
            fig.colorbar(CS, ax=ax)
        plt.tight_layout()
        plt.savefig('figs/deviation_sub_{}.png'.format(metric))


if __name__ == '__main__':

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

    METRICS = ['T1GRS', 'T2GRS']

    for metric in METRICS:
        plot_bootstrap_results(metric)
