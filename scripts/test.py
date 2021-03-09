#!/usr/bin/env python3

import os
import sys
import pathlib

import pytest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
print(sys.path)
nproc = os.cpu_count()
print(f"nproc = {nproc}")

import dpe.datasets as ds
import dpe


seed = 0
# np.random.seed(seed)

p_C = 0.1
# scores = ds.generate_dataset(p_C=p_C)

# @pytest.fixture
def scores_pC010():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=5000, seed=seed)

nproc = 8
scores = scores_pC010()
n_boot = 10
n_mix = 10
summary, bootstraps = dpe.analyse_mixture(scores, n_boot=n_boot, n_mix=n_mix, seed=seed, n_jobs=nproc, true_pC=p_C)
print(summary)
