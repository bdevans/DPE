#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:18:50 2019

@author: ben
"""

import time
import os

import numpy as np
import pandas as pd

import proportion_estimation as pe
import datasets as ds

out_dir = 'variability_data'
out_dir = os.path.join("results", "variability_data")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

seed = 42
p_D = 0.5
n_boot = 1000
sample_size = 1000  # -1
n_mix = 100
alpha = 0.05
methods = None

datasets = {}
datasets["Renal"] = ds.load_renal_data()
datasets["Diabetes"] = ds.load_diabetes_data("T1GRS")

for tag, data in datasets.items():

    scores, bins, means, medians, _ = data
    Ref_D, Ref_C = scores["Ref1"], scores["Ref2"]
    assert(0.0 <= p_D <= 1.0)
    n_D = int(round(sample_size * p_D))
    n_C = sample_size - n_D

    # Construct mixture
    mixture = np.concatenate((np.random.choice(Ref_D, n_D, replace=True),
                              np.random.choice(Ref_C, n_C, replace=True)))
    np.savetxt(os.path.join(out_dir, "{}_mixture.csv".format(tag)), mixture)
    scores["Mix"] = mixture

    t = time.time()  # Start timer
    results = pe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                 boot_size=sample_size, n_mix=n_mix, alpha=alpha,
                                 true_p1=p_D, n_jobs=-1, seed=seed, verbose=1)

    results.to_csv(os.path.join(out_dir, "{}_bootstraps.csv".format(tag)), header=True)
    elapsed = time.time() - t
    print('Elapsed time = {:.3f} seconds\n'.format(elapsed))
