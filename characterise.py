#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:28 2018

Parallelised script for systematically characterising the parameter space for
proportion estimation methods by generating artificial mixtures across a grid
of proprtions and sample sizes.

@author: Benjamin D. Evans
"""

import os
import time

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

import proportion_estimation as pe
import datasets as ds

# ---------------------------- Define constants ------------------------------

verbose = False
seed = 42

n_samples = 1000  # 1000
n_boot = 0
n_mix = 0
alpha = 0.05
# sample_sizes = np.arange(100, 1001, 100)  # 3100
sample_sizes = np.linspace(100, 2500, 25, endpoint=True, dtype=int)
# proportions = np.arange(0.0, 1.01, 0.01)  # Ref1 propoertions
proportions = np.linspace(0.0, 1.0, 101, endpoint=True)

# Very quick testing run
# n_samples = 4  # 1000
# n_boot = 4
# sample_sizes = np.arange(100, 201, 100)
# proportions = np.linspace(0.0, 1.0, 6, endpoint=True)

# KDE_kernel = 'gaussian'
# kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

out_dir = os.path.join("results", "s{}_m{}_b{}".format(n_samples, n_mix, n_boot))

# ----------------------------------------------------------------------------

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if verbose and not os.path.exists(os.path.join(out_dir, "logs")):
    os.makedirs(os.path.join(out_dir, "logs"))


def SecToStr(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u'%d:%02d:%02d' % (h, m, s)


def assess(sample_size, pC, RefC, RefN, bins, methods, n_boot, seed=None):
    '''New method using analyse_mixture'''

    assert(0.0 <= pC <= 1.0)
    n_RefC = int(round(sample_size * pC))
    n_RefN = sample_size - n_RefC

    # Construct mixture
    mixture = np.concatenate((np.random.choice(RefC, n_RefC, replace=True),
                              np.random.choice(RefN, n_RefN, replace=True)))

    scores = {'Ref1': RefC, 'Ref2': RefN}
    scores['Mix'] = mixture

    if verbose:
        logfile = os.path.join(out_dir, "logs", 'pe_s{:05}_p{:.2f}_{}.log'
                                                .format(sample_size, pC, seed))
    else:
        logfile = None

    results = pe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                 boot_size=-1, n_mix=n_mix, alpha=alpha,
                                 true_p1=pC, n_jobs=1, seed=seed,
                                 verbose=0, logfile=logfile)
    point = results.iloc[[0]]

    if n_boot or n_mix:
        boots = results.iloc[1:, :]
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
        scores, bins, means, medians, pC = data
        RefC = scores['Ref1']
        RefN = scores['Ref2']
        Mix = scores['Mix']
        bin_width = bins['width']
        bin_edges = bins['edges']

        methods = pe.prepare_methods(scores, bins)  # Get all methods

        # NOTE: There is always a point estimate in addition to any bootstraps
        n_applications = len(sample_sizes) * len(proportions) * n_samples * (1+n_boot)
        print("Total method applications (of {} methods): {:,}"
              .format(len(methods), n_applications))

        with open(os.path.join(out_dir, "run.log"), "a") as runlog:
            runlog.write("Running parameter sweep with {} scores...\n".format(tag))
            runlog.write("Samples sizes: {:,}; Proportions: {:,}; Mixtures: {:,}; Bootstraps: {:,}\n"
                         .format(len(sample_sizes), len(proportions), n_samples, n_boot))
            runlog.write("Total method applications (of {} methods): {:,}\n"
                         .format(len(methods), n_applications))

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
            for p, pC in enumerate(prop_bar):
                prop_bar.set_description(" p1* = {:6.2f}".format(pC))

                # Make mixtures deterministic with parallelism
                # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
                mix_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_samples)

                # Spawn threads
                with Parallel(n_jobs=nprocs) as parallel:
                    # Parallelise over mixtures
                    results = parallel(delayed(assess)(sample_size, pC,
                                                       RefC, RefN, bins, methods,
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
            np.save('{}/point_{}_{}'.format(out_dir, method, tag),
                    point_arrays[method])
            if n_boot:
                np.save('{}/boots_{}_{}'.format(out_dir, method, tag),
                        boots_arrays[method])
        # TODO: Save the arrays in the dictionary as a pickle file
        np.save('{}/sample_sizes_{}'.format(out_dir, tag), sample_sizes)
        np.save('{}/proportions_{}'.format(out_dir, tag), proportions)

    print("Analysis of methods on datasets: {} complete!".format(list(datasets)))
    with open(os.path.join(out_dir, "run.log"), "a") as runlog:
        runlog.write("Analysis of methods on datasets: {} complete!\n".format(list(datasets)))
        runlog.write(time.strftime("%Y-%m-%d (%a) %H:%M:%S\n", time.gmtime()))
