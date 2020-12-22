#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:14:30 2018

@author: ben
"""

import time

import numpy as np
import proportion_estimation as pe
from datasets import (load_diabetes_data, load_renal_data)

out_dir = 'benchmark'
seed = 42
n_mix = 5
bootstraps = 100
boot_size = -1  #1000  # -1
alpha = 0.05
CI_METHOD = "experimental"  #"jeffreys"
KDE_kernel = 'gaussian'

# (scores, bins, means, medians, p1_star) = load_renal_data()
(scores, bins, means, medians, p1_star) = load_diabetes_data("T1GRS")
methods = None

# Multicore
t = time.time()  # Start timer
results = pe.analyse_mixture(scores, bins, methods, n_boot=bootstraps, n_mix=n_mix,
                             boot_size=boot_size, alpha=alpha, true_p1=p1_star,
                             n_jobs=-1, seed=seed, verbose=1)

print(results.head())
elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds\n'.format(elapsed))

# Single core
t = time.time()  # Start timer
results = pe.analyse_mixture(scores, bins, methods, n_boot=bootstraps, n_mix=n_mix,
                             boot_size=boot_size, alpha=alpha, true_p1=p1_star,
                             n_jobs=1, seed=seed, verbose=1)

print(results.head())
elapsed = time.time() - t
print('Elapsed time = {:.3f} seconds\n'.format(elapsed))
