#!/usr/bin/env python

# Run this script from the command line as an argument to the profiler:
# python -m cProfile -s cumtime profile_analysis.py

import os

from dpe import analyse_mixture, fit_kernel
from dpe.datasets import load_diabetes_data


seed = 42
n_boot = 10
sample_size = 1000  # -1
n_mix = 10
alpha = 0.05
methods = 'all'
n_jobs = 1
ci_method = "bca"
correct_bias = False
kernel = "gaussian"
n_head = 50

out_dir = os.path.join("results", "profile")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

scores, bins, means, medians, p_C = load_diabetes_data("T1GRS")


if __name__ == "__main__":
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    # results = analyse_mixture(scores, bins, methods, n_boot=n_boot,
    #                     boot_size=sample_size, n_mix=n_mix, alpha=alpha,
    #                     true_pC=p_C, n_jobs=n_jobs, seed=seed, verbose=1,
    #                     ci_method=ci_method, correct_bias=correct_bias,
    #                     logfile=os.path.join(out_dir, "profile.log"))

    kde_mix = fit_kernel(scores["Mix"], bw=bins['width'], kernel=kernel)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.strip_dirs()
    stats.print_stats(n_head)
