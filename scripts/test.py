#!/usr/bin/env python3

import os
import sys
import pathlib
import pprint

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

# scores = ds.generate_dataset(p_C=p_C)

# @pytest.fixture
def scores_pC010_size100():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=100, seed=seed)

# @pytest.fixture
def scores_pC020_size500():
    """Generate synthetic data with p_C=0.2"""
    return ds.generate_dataset(p_C=0.2, size=500, seed=seed)

p_C = 0.1
scores = scores_pC010_size100()
# p_C = 0.2
# scores = scores_pC020_size500()
n_boot = 10
n_mix = 10
nproc = 8
summary, bootstraps = dpe.analyse_mixture(scores, n_boot=n_boot, n_mix=n_mix, seed=seed, n_jobs=nproc, true_pC=p_C)
print(f"{n_boot=}, {n_mix=}, {seed=}, {nproc=}")
# pprint.pprint(summary)
print(summary)
