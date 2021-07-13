#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import sys
import pathlib

project_dir = str(pathlib.Path(__file__).parent.parent.absolute())
if project_dir not in sys.path:
    print(f"Adding {project_dir} to PYTHONPATH...")
    sys.path.insert(0, project_dir)

import dpe
import dpe.datasets as ds


# Synthetic data ground truth proportion of cases (p_C) in the mixtures
example_proportions = [0.25, 0.50, 0.75]  # List of p_C values

# Main parameters to edit
n_mix = 100  # Number of mixtures to create based on the initial point estimate
n_boot = 100  # 1000  # Number of bootstraps to create (for each mixture)
load_data = True  # Change this to False to generate new datasets

# Less frequently changed parameters
seed = 42  # Set this to seed the PRNG for reproducibility of results
n_jobs = -1  # Change this to set the number of CPUs to use in parallel or -1 to use all
ci_method = "bca"  # Method for calculating confidence intervals: Bias-corrected and accelerated bootstrap interval
correct_bias = True  # Flag to apply bootstrap correction to the estimate
boot_size = -1  # Size of each bootstrap (set to -1 to make it the same size as the mixture distribution)
alpha = 0.05  # Used to set the centile for calculating the confidence intervals from bootstrapped estimates
bins = 'fd'  # Use Freedman-Diaconis rule for binning data by default
methods = 'all'  # ["Excess", "Means", "EMD", "KDE"]  # Alternatively pass a list of chosen methods
verbose = 1  # Set the level of console output verbosity (0, 1, 2, 3. -1 : Progress bars only)


print('=' * 80)
for p_C in example_proportions:

    dataset = f'example_pC{int(round(100*p_C)):03}'
    print(f'Running on example dataset: p_C = {p_C}')
    if load_data:
        # Load the example dataset
        print(f'Loading dataset: {dataset}...')
        scores = ds.load_dataset(f'{dataset}.csv')
    else:
        # Generate new data
        print('Generating fresh example dataset...')
        scores = ds.generate_dataset(p_C)

        # Save data
        ds.save_dataset(scores, f'{dataset}.csv')

    # Create output directory
    out_dir = os.path.join("results", dataset)
    os.makedirs(out_dir, exist_ok=True)

    t = time.time()  # Start timer
    # NOTE: the ground truth p_C is passed to analyse_mixture but is only used
    # for printing in the log file for easy comparison and may be omitted
    results = dpe.analyse_mixture(scores, bins, methods, n_boot=n_boot,
                                  boot_size=boot_size, n_mix=n_mix, alpha=alpha,
                                  ci_method=ci_method, correct_bias=correct_bias,
                                  seed=seed, n_jobs=n_jobs, verbose=verbose,
                                  true_pC=p_C, logfile=f'{dataset}.log')

    summary, bootstraps = results
    bootstraps.to_csv(os.path.join(out_dir, f"{dataset}_bootstraps.csv"), header=True)
    elapsed = time.time() - t
    print(f'Elapsed time = {elapsed:.3f} seconds')
    print('=' * 80, '\n')
