#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

Parallelised script for systematically characterising the parameter space for
proportion estimation methods by generating artificial mixtures across a grid
of proprtions and sample sizes.

@author: ben

Original Diabetes data was processed with commit 01a9705b
https://git.exeter.ac.uk/bdevans/DPE/commit/01a9705b6fa1bf0d1df4fd3a4beaa1a413f64cb5
"""

import os
import time

import numpy as np
from tqdm import tqdm
# from joblib import Memory
# mem = Memory(cachedir='/tmp')
from joblib import Parallel, delayed, cpu_count

import proportion_estimation as pe
import datasets as ds
# from datasets import (load_diabetes_data, load_renal_data)

# ---------------------------- Define constants ------------------------------

out_dir = "results"
verbose = False
seed = 42

n_samples = 1000  # 1000
n_boot = 0
sample_sizes = np.arange(100, 2501, 100)  # 3100
# sample_sizes = np.linspace(100, 2500, 25, endpoint=True, dtype=int)
# proportions = np.arange(0.0, 1.01, 0.01)  # Ref1 propoertions
proportions = np.linspace(0.0, 1.0, 101, endpoint=True)

# Very quick testing run
n_samples = 4  # 1000
n_boot = 4
sample_sizes = np.arange(100, 201, 100)
proportions = np.linspace(0.0, 1.0, 6, endpoint=True)

KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

# out_dir = "results_m{}_b{}".format(n_samples, n_boot)

# ----------------------------------------------------------------------------

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def SecToStr(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u'%d:%02d:%02d' % (h, m, s)


def assess(sample_size, prop_Ref1, Ref1, Ref2, bins, methods, n_boot, seed=None):
    '''New method using analyse_mixture'''

    assert(0.0 <= prop_Ref1 <= 1.0)
    n_Ref1 = int(round(sample_size * prop_Ref1))
    n_Ref2 = sample_size - n_Ref1

    # Construct mixture
    mixture = np.concatenate((np.random.choice(Ref1, n_Ref1, replace=True),
                              np.random.choice(Ref2, n_Ref2, replace=True)))

    scores = {'Ref1': Ref1, 'Ref2': Ref2}
    scores['Mix'] = mixture

    point = pe.analyse_mixture(scores, bins, methods, n_boot=0,
                               boot_size=-1, alpha=0.05,
                               true_p1=prop_Ref1, n_jobs=1, seed=seed,
                               verbose=0, logfile=None)

    if n_boot:
        logfile = 'pe_s{}_p{}.log'.format(sample_size, prop_Ref1)
        boots = pe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                   boot_size=-1, alpha=0.05,
                                   true_p1=prop_Ref1, n_jobs=1, seed=seed,
                                   verbose=0, logfile=logfile)
    else:
        boots = {method: None for method in methods}

    return (point, boots)


if __name__ == '__main__':

    nprocs = cpu_count()
    print('Running with {} processors...'.format(nprocs))

    with open(os.path.join(out_dir, "run.log"), "w") as runlog:
        runlog.write(time.strftime("%Y-%m-%d (%a) %H:%M:%S\n", time.gmtime()))
        runlog.write("=========================\n\n")
        runlog.write("Random seed: {}\n".format(seed))
        runlog.write("Saving results to: {}\n".format(out_dir))
        runlog.write("Running with {} processors...\n\n".format(nprocs))

    # Set random seed
    np.random.seed(seed)

    np.seterr(divide='ignore', invalid='ignore')

    datasets = {}
    datasets["Renal"] = ds.load_renal_data()
    datasets["Diabetes"] = ds.load_diabetes_data("T1GRS")

    for tag, data in datasets.items():

        print("Running parameter sweep with {} scores...".format(tag))
        print("Samples sizes: {:,}; Proportions: {:,}; Mixtures: {:,}; Bootstraps: {:,}"
              .format(len(sample_sizes), len(proportions), n_samples, n_boot))

        t = time.time()  # Start timer

        # Unpack data
        scores, bins, means, medians, prop_Ref1 = data
        Ref1 = scores['Ref1']
        Ref2 = scores['Ref2']
        Mix = scores['Mix']
        bin_width = bins['width']
        bin_edges = bins['edges']

        methods = pe.prepare_methods_(scores, bins)  # Get all methods

        print("Total method applications (of {} methods): {:,}"
              .format(len(methods), len(sample_sizes)*len(proportions)*n_samples*n_boot))

        with open(os.path.join(out_dir, "run.log"), "a") as runlog:
            runlog.write("Running parameter sweep with {} scores...\n".format(tag))
            runlog.write("Samples sizes: {:,}; Proportions: {:,}; Mixtures: {:,}; Bootstraps: {:,}\n"
                         .format(len(sample_sizes), len(proportions), n_samples, n_boot))
            runlog.write("Total method applications (of {} methods): {:,}\n"
                         .format(len(methods), len(sample_sizes)*len(proportions)*n_samples*n_boot))

        point_arrays = {}
        boots_arrays = {}
        for method in methods:
            point_arrays[method] = np.zeros((len(sample_sizes),
                                             len(proportions),
                                             n_samples))
            boots_arrays[method] = np.zeros((len(sample_sizes),
                                             len(proportions),
                                             n_samples, n_boot))

        size_bar = tqdm(sample_sizes, dynamic_ncols=True)
        for s, sample_size in enumerate(size_bar):
            size_bar.set_description("Size = {:6,}".format(sample_size))

            prop_bar = tqdm(proportions, dynamic_ncols=True)
            for p, prop_Ref1 in enumerate(prop_bar):
                prop_bar.set_description(" p1* = {:6.2f}".format(prop_Ref1))

                # Make mixtures deterministic with parallelism
                # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
                mix_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_samples)

                # Spawn threads
                with Parallel(n_jobs=nprocs) as parallel:
                    # Parallelise over mixtures
                    results = parallel(delayed(assess)(sample_size, prop_Ref1,
                                                       Ref1, Ref2, bins, methods,
                                                       n_boot, seed=seed)

                                       for seed in tqdm(mix_seeds,
                                                        desc=" Mixture     ",
                                                        dynamic_ncols=True))

                    for m in range(n_samples):
                        point, boots = results[m]
                        for method in methods:
                            point_arrays[method][s, p, m] = point[method]
                            if n_boot:
                                boots_arrays[method][s, p, m, :] = boots[method]

        elapsed = time.time() - t
        print('Elapsed time = {}\n'.format(SecToStr(elapsed)))
        with open(os.path.join(out_dir, "run.log"), "a") as runlog:
            runlog.write('Elapsed time = {}\n'.format(SecToStr(elapsed)))
            runlog.write("=======================\n\n")

        # Normalise by EMD 1<->2 (EMD distance between the two orignal distributions)
#        if run_EMD:
#            norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
#            norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21
#            if check_EMD:
#                norm_EMD_dev = emd_dev_from_fit * bin_width / max_emd / i_EMD_21
#                median_error = 100 * np.median(norm_EMD_dev, axis=2)  # Percentage
        # TODO: combine and pickle results
        results_arrays = {}

        for method in methods:
            np.save('{}/point_{}_{}'.format(out_dir, method, tag), point_arrays[method])
            if n_boot:
                np.save('{}/boots_{}_{}'.format(out_dir, method, tag), boots_arrays[method])
        # TODO: Save the arrays in the dictionary as a pickle file
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    print("Analysis of methods on datasets: {} complete!".format(list(datasets)))
    with open(os.path.join(out_dir, "run.log"), "a") as runlog:
        runlog.write("Analysis of methods on datasets: {} complete!\n".format(list(datasets)))
        runlog.write(time.strftime("%Y-%m-%d (%a) %H:%M:%S\n", time.gmtime()))
