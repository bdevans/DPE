#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:14:30 2018

@author: ben
"""

import time

import dpe
import dpe.datasets as ds

out_dir = 'benchmark'
seed = 42
n_mix = 5
bootstraps = 100
boot_size = -1  # 1000  # -1
alpha = 0.05
CI_METHOD = "bca"  # "experimental"  # "jeffreys"
KDE_kernel = 'gaussian'

# (scores, bins, means, medians, p1_star) = ds.load_renal_data()
(scores, bins, means, medians, p1_star) = ds.load_diabetes_data("T1GRS")
methods = None

# Multicore
t = time.time()  # Start timer
results = dpe.analyse_mixture(scores, bins, methods, n_boot=bootstraps, n_mix=n_mix,
                              boot_size=boot_size, alpha=alpha, true_p1=p1_star,
                              n_jobs=-1, seed=seed, verbose=1)

print(results[1].head())
elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

# Single core
t = time.time()  # Start timer
results = dpe.analyse_mixture(scores, bins, methods, n_boot=bootstraps, n_mix=n_mix,
                              boot_size=boot_size, alpha=alpha, true_p1=p1_star,
                              n_jobs=1, seed=seed, verbose=1)

print(results[1].head())
elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds\n'.format(elapsed))
