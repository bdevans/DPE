import os
import sys
# import pathlib

import pytest
import numpy as np

# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
# print(sys.path)
nproc = os.cpu_count()

import dpe.datasets as ds
import dpe


seed = 0
p_C = 0.1

@pytest.fixture
def scores_p010():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=5000, seed=seed)


def test_anaylse_mixture(scores_p010):
    results = dpe.analyse_mixture(scores_p010, n_boot=0, n_mix=0, seed=seed, true_pC=p_C)
    summary, bootstraps = results

    assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
    assert np.isclose(summary["Excess"]["p_C"], 0.0964)
    assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
    assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)


# def test_anaylse_mixture_mix(scores_p010):
#     results = dpe.analyse_mixture(scores_p010, n_boot=0, n_mix=10, seed=seed, true_pC=p_C)
#     summary, bootstraps = results

#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)


def test_anaylse_mixture_boot(scores_p010):
    results = dpe.analyse_mixture(scores_p010, n_boot=10, n_mix=0, seed=seed, true_pC=p_C)
    summary, bootstraps = results

    # Proportion estimates
    assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
    assert np.isclose(summary["Excess"]["p_C"], 0.0964)
    assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
    assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

    # Bootstrap estimates
    assert np.isclose(summary["EMD"]["mean"], 0.12269521263948222), summary["EMD"]["mean"]
    assert np.isclose(summary["EMD"]["std"], 0.013433323691734361), summary["EMD"]["std"]
    assert np.isclose(summary["EMD"]["CI"][0], 0.11028732209621961), summary["EMD"]["CI"][0]
    assert np.isclose(summary["EMD"]["CI"][1], 0.14805938902974064), summary["EMD"]["CI"][1]

    assert np.isclose(summary["Excess"]["mean"], 0.10216)
    assert np.isclose(summary["Excess"]["std"], 0.01102027222894244)
    assert np.isclose(summary["Excess"]["CI"][0], 0.0952)
    assert np.isclose(summary["Excess"]["CI"][1], 0.1248)

    assert np.isclose(summary["KDE"]["mean"], 0.12278166955266238)
    assert np.isclose(summary["KDE"]["std"], 0.012485646023580961)
    assert np.isclose(summary["KDE"]["CI"][0], 0.10882118268399207)
    assert np.isclose(summary["KDE"]["CI"][1], 0.1448788902735669)

    assert np.isclose(summary["Means"]["mean"], 0.12279898550753088)
    assert np.isclose(summary["Means"]["std"], 0.01336121417716939)
    assert np.isclose(summary["Means"]["CI"][0], 0.11085840202841786)
    assert np.isclose(summary["Means"]["CI"][1], 0.14815690928847491)


def test_anaylse_mixture_boot_parallel(scores_p010):
    results = dpe.analyse_mixture(scores_p010, n_boot=10, n_mix=0, seed=seed, n_jobs=nproc, true_pC=p_C)
    summary, bootstraps = results

    # Proportion estimates
    assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
    assert np.isclose(summary["Excess"]["p_C"], 0.0964)
    assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
    assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

    # Bootstrap estimates
    # NOTE: These are slightly different than when n_jobs==1
    assert np.isclose(summary["EMD"]["mean"], 0.12267561935887468), summary["EMD"]["mean"]
    assert np.isclose(summary["EMD"]["std"], 0.00860512889035671), summary["EMD"]["std"]
    assert np.isclose(summary["EMD"]["CI"][0], 0.10911018704388409), summary["EMD"]["CI"][0]
    assert np.isclose(summary["EMD"]["CI"][1], 0.13124582773285004), summary["EMD"]["CI"][1]

    assert np.isclose(summary["Excess"]["mean"], 0.09759999999999999), summary["Excess"]["mean"]
    assert np.isclose(summary["Excess"]["std"], 0.0100860299424501), summary["Excess"]["std"]
    assert np.isclose(summary["Excess"]["CI"][0], 0.0776), summary["Excess"]["CI"][0]
    assert np.isclose(summary["Excess"]["CI"][1], 0.1108), summary["Excess"]["CI"][1]

    assert np.isclose(summary["KDE"]["mean"], 0.11523613740188092), summary["KDE"]["mean"]
    assert np.isclose(summary["KDE"]["std"], 0.011402856318140971), summary["KDE"]["std"]
    assert np.isclose(summary["KDE"]["CI"][0], 0.09797229341051193), summary["KDE"]["CI"][0]
    assert np.isclose(summary["KDE"]["CI"][1], 0.13136957654751658), summary["KDE"]["CI"][1]

    assert np.isclose(summary["Means"]["mean"], 0.12283908114071776), summary["Means"]["mean"]
    assert np.isclose(summary["Means"]["std"], 0.008544651916214157), summary["Means"]["std"]
    assert np.isclose(summary["Means"]["CI"][0], 0.11014270881728092), summary["Means"]["CI"][0]
    assert np.isclose(summary["Means"]["CI"][1], 0.13095145348487366), summary["Means"]["CI"][1]


def test_anaylse_mixture_mix_boot(scores_p010):
    results = dpe.analyse_mixture(scores_p010, n_boot=10, n_mix=10, seed=seed, true_pC=p_C)
    summary, bootstraps = results

    # Proportion estimates
    assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
    assert np.isclose(summary["Excess"]["p_C"], 0.0964)
    assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
    assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

    # Bootstrap estimates
    assert np.isclose(summary["EMD"]["mean"], 0.11991567686380979)
    assert np.isclose(summary["EMD"]["std"], 0.01879614372677909)
    assert np.isclose(summary["EMD"]["CI"][0], 0.0835170826819473)
    assert np.isclose(summary["EMD"]["CI"][1], 0.15547233358398993)

    assert np.isclose(summary["Excess"]["mean"], 0.08225199999999999)
    assert np.isclose(summary["Excess"]["std"], 0.016321240639118093)
    assert np.isclose(summary["Excess"]["CI"][0], 0.0532)
    assert np.isclose(summary["Excess"]["CI"][1], 0.1176)

    assert np.isclose(summary["KDE"]["mean"], 0.1180030865344234)
    assert np.isclose(summary["KDE"]["std"], 0.017439527912894638)
    assert np.isclose(summary["KDE"]["CI"][0], 0.09177886669333177)
    assert np.isclose(summary["KDE"]["CI"][1], 0.15453785682429844)

    assert np.isclose(summary["Means"]["mean"], 0.10913808007686349)
    assert np.isclose(summary["Means"]["std"], 0.018587226882665266)
    assert np.isclose(summary["Means"]["CI"][0], 0.0719092323672722)
    assert np.isclose(summary["Means"]["CI"][1], 0.1400325155522334)


def test_anaylse_mixture_mix_boot_parallel(scores_p010):
    results = dpe.analyse_mixture(scores_p010, n_boot=10, n_mix=10, seed=seed, n_jobs=nproc, true_pC=p_C)
    summary, bootstraps = results

    # Proportion estimates
    assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
    assert np.isclose(summary["Excess"]["p_C"], 0.0964)
    assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
    assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

    # Bootstrap estimates
    assert np.isclose(summary["EMD"]["mean"], 0.11991567686380979)
    assert np.isclose(summary["EMD"]["std"], 0.01879614372677909)
    assert np.isclose(summary["EMD"]["CI"][0], 0.0835170826819473)
    assert np.isclose(summary["EMD"]["CI"][1], 0.15547233358398993)

    assert np.isclose(summary["Excess"]["mean"], 0.08225199999999999)
    assert np.isclose(summary["Excess"]["std"], 0.016321240639118093)
    assert np.isclose(summary["Excess"]["CI"][0], 0.0532)
    assert np.isclose(summary["Excess"]["CI"][1], 0.1176)

    assert np.isclose(summary["KDE"]["mean"], 0.1180030865344234)
    assert np.isclose(summary["KDE"]["std"], 0.017439527912894638)
    assert np.isclose(summary["KDE"]["CI"][0], 0.09177886669333177)
    assert np.isclose(summary["KDE"]["CI"][1], 0.15453785682429844)

    assert np.isclose(summary["Means"]["mean"], 0.10913808007686349)
    assert np.isclose(summary["Means"]["std"], 0.018587226882665266)
    assert np.isclose(summary["Means"]["CI"][0], 0.0719092323672722)
    assert np.isclose(summary["Means"]["CI"][1], 0.1400325155522334)
