#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:18:50 2019

@author: ben
"""

import time
import os

import numpy as np

import dpe
import dpe.datasets as ds

out_dir = os.path.join("results", "variability_data")
os.makedirs(out_dir, exist_ok=True)

seed = 42
p_C = 0.5
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
    R_C, R_N = scores["R_C"], scores["R_N"]
    assert(0.0 <= p_C <= 1.0)
    n_C = int(round(sample_size * p_C))
    n_N = sample_size - n_C

    # Construct mixture
    mixture = np.concatenate((np.random.choice(R_C, n_C, replace=True),
                              np.random.choice(R_N, n_N, replace=True)))
    np.savetxt(os.path.join(out_dir, "{}_mixture.csv".format(tag)), mixture)
    scores["Mix"] = mixture

    t = time.time()  # Start timer
    results = dpe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                  boot_size=sample_size, n_mix=n_mix, alpha=alpha,
                                  true_pC=p_C, n_jobs=-1, seed=seed, verbose=1)

    summary, samples = results
    samples.to_csv(os.path.join(out_dir, "{}_bootstraps.csv".format(tag)), header=True)
    elapsed = time.time() - t
    print('Elapsed time = {:.3f} seconds\n'.format(elapsed))
