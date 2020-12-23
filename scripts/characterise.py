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

# NOTE: Add the module path to sys.path if calling from the scripts subdirectory
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

import dpe
import dpe.datasets as ds
from dpe.utilities import construct_mixture, format_seconds


# ---------------------------- Define constants ------------------------------

verbose = False
seed = 42

n_samples = 1000  # 1000
n_boot = 0
n_mix = 0
alpha = 0.05
# sample_sizes = np.arange(100, 1001, 100)  # 3100
sample_sizes = np.linspace(100, 2500, 25, endpoint=True, dtype=int)
# proportions = np.arange(0.0, 1.01, 0.01)  # R_C propoertions
proportions = np.linspace(0.0, 1.0, 101, endpoint=True)

# Very quick testing run
# n_samples = 4  # 1000
# n_boot = 4
# sample_sizes = np.arange(100, 201, 100)
# proportions = np.linspace(0.0, 1.0, 6, endpoint=True)

out_dir = os.path.join("results", f"s{n_samples}_m{n_mix}_b{n_boot}")

# ----------------------------------------------------------------------------

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if verbose and not os.path.exists(os.path.join(out_dir, "logs")):
    os.makedirs(os.path.join(out_dir, "logs"))


def assess(sample_size, p_C, R_C, R_N, bins, methods, n_boot, seed=None):
    '''New method using analyse_mixture'''

    mixture = construct_mixture(R_C, R_N, p_C, sample_size)

    scores = {'R_C': R_C, 'R_N': R_N}
    scores['Mix'] = mixture

    if verbose:
        logfile = os.path.join(out_dir, "logs", 
                               f'dpe_s{sample_size:05}_p{p_C:.2f}_{seed}.log')
    else:
        logfile = None

    results = dpe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                boot_size=-1, n_mix=n_mix, alpha=alpha,
                                true_pC=p_C, n_jobs=1, seed=seed,
                                verbose=0, logfile=logfile)
    summary, samples = results
    point = samples.iloc[[0]]

    if n_boot or n_mix:
        boots = samples.iloc[1:, :]
    else:
        boots = {method: None for method in methods}

    return (point, boots)


if __name__ == '__main__':

    nprocs = cpu_count()
    print(f'Running with {nprocs} processors...')

    with open(os.path.join(out_dir, "run.log"), "w") as runlog:
        runlog.write(time.strftime("%Y-%m-%d (%a) %H:%M:%S\n", time.gmtime()))
        runlog.write("=========================\n\n")
        runlog.write(f"Random seed: {seed}\n")
        runlog.write(f"Saving results to: {out_dir}\n")
        runlog.write(f"Running with {nprocs} processors...\n\n")

    # Set random seed
    np.random.seed(seed)

    np.seterr(divide='ignore', invalid='ignore')

    datasets = {}
    datasets["Renal"] = ds.load_renal_data()
    datasets["Diabetes"] = ds.load_diabetes_data("T1GRS")

    for tag, data in datasets.items():

        print(f"Running parameter sweep with {tag} scores...")
        print(f"Samples sizes: {len(sample_sizes):,}; Proportions: {len(proportions):,}; Mixtures: {n_samples:,}; Bootstraps: {n_boot:,}")

        t = time.time()  # Start timer

        # Unpack data
        scores, bins, means, medians, p_C = data
        R_C = scores['R_C']
        R_N = scores['R_N']
        Mix = scores['Mix']
        bin_width = bins['width']
        bin_edges = bins['edges']

        methods = dpe.prepare_methods(scores, bins)  # Get all methods

        # NOTE: There is always a point estimate in addition to any bootstraps
        n_applications = len(sample_sizes) * len(proportions) * n_samples * (1+n_boot)
        print(f"Total method applications (of {len(methods)} methods): {n_applications:,}")

        with open(os.path.join(out_dir, "run.log"), "a") as runlog:
            runlog.write(f"Running parameter sweep with {tag} scores...\n")
            runlog.write(f"Samples sizes: {len(sample_sizes):,}; Proportions: {len(proportions):,}; Mixtures: {n_samples:,}; Bootstraps: {n_boot:,}\n")
            runlog.write(f"Total method applications (of {len(methods)} methods): {n_applications:,}\n")

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
            size_bar.set_description(f"Size = {sample_size:6,}")

            prop_bar = tqdm(proportions, dynamic_ncols=True)
            for p, p_C in enumerate(prop_bar):
                prop_bar.set_description(f" p_C = {p_C:6.2f}")

                # Make mixtures deterministic with parallelism
                # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
                mix_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_samples)

                # Spawn threads
                with Parallel(n_jobs=nprocs) as parallel:
                    # Parallelise over mixtures
                    results = parallel(delayed(assess)(sample_size, p_C,
                                                       R_C, R_N, bins, methods,
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
        print(f'Elapsed time = {format_seconds(elapsed)}\n')
        with open(os.path.join(out_dir, "run.log"), "a") as runlog:
            runlog.write(f'Elapsed time = {format_seconds(elapsed)}\n')
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
            np.save(os.path.join(out_dir, f"point_{method}_{tag}"), point_arrays[method])
            if n_boot:
                np.save(os.path.join(out_dir, f"boots_{method}_{tag}"), boots_arrays[method])
        # TODO: Save the arrays in the dictionary as a pickle file
        np.save(os.path.join(out_dir, f"sample_sizes_{tag}"), sample_sizes)
        np.save(os.path.join(out_dir, f"proportions_{tag}"), proportions)

    print(f"Analysis of methods on datasets: {list(datasets)} complete!")
    with open(os.path.join(out_dir, "run.log"), "a") as runlog:
        runlog.write(f"Analysis of methods on datasets: {list(datasets)} complete!\n")
        runlog.write(time.strftime("%Y-%m-%d (%a) %H:%M:%S\n", time.gmtime()))
