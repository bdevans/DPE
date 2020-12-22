#!/usr/bin/env python

# Script to benchmark performance of the KDE method
# There are two main computational costs
# 1. Convolving kernels with the data. 
# This depends upon: the kernel, number of points and the algorithm
# 2. The least-squares fitting algorithm

import time
import timeit
import os
# import numpy as np

from datasets import load_diabetes_data
from proportion_estimation import analyse_mixture
from proportion_estimation import fit_kernel


scores, bins, means, medians, p_C = load_diabetes_data('T1GRS')
kernel = "gaussian"



time_skl = timeit.timeit(stmt="fit_kernel(scores['Mix'], bw=bins['width'], kernel=kernel)", 
              setup="from datasets import load_diabetes_data; from proportion_estimation import fit_kernel; scores, bins, means, medians, p_C, kernel = *load_diabetes_data('T1GRS'), 'gaussian'",
              number=1000,
              timer=time.process_time)
            #   timer=time.time)

# print(np.amin(time_skl))
print(time_skl)


# fit_kernel(scores['Mix'], bw=bins['width'], kernel=kernel)

# time_skl = timeit.timeit(stmt="fit_kernel(scores['Mix'], bw=bins['width'], kernel=kernel)", 
#               setup="from datasets import load_diabetes_data; from proportion_estimation import fit_kernel; scores, bins, means, medians, p_C, kernel = *load_diabetes_data('T1GRS'), 'gaussian'",
#               number=10,
#               timer=time.process_time)

