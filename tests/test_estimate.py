import os

import pytest
import numpy as np

import dpe.datasets as ds
import dpe


seed = 0
# p_C = 0.1
nproc = os.cpu_count()


@pytest.fixture
def scores_pC010():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=5000, seed=seed)


expected_point = {
    'Excess': {'p_C': 0.0964},
    'Means': {'p_C': 0.12116137064813201},
    'EMD': {'p_C': 0.12156986188389857},
    'KDE': {'p_C': 0.11653533406952703}
    }

expected_n_boot_10 = {
    'Excess': {'p_C': 0.0964, 'CI': (0.0788, 0.11), 'mean': 0.09623999999999999, 'std': 0.013116340953177453},
    'Means': {'p_C': 0.12116137064813201, 'CI': (0.10567834935848158, 0.13408197987526282), 'mean': 0.12062971258005899, 'std': 0.010329231195737199},
    'EMD': {'p_C': 0.12156986188389857, 'CI': (0.10695540862358027, 0.13506864157907), 'mean': 0.12130475132716852, 'std': 0.010077862989318908},
    'KDE': {'p_C': 0.11653533406952703, 'CI': (0.10008488391610758, 0.13113053107870062), 'mean': 0.11330220847636982, 'std': 0.011794233996819635}
    }

expected_n_boot_10_parallel = {
    'Excess': {'p_C': 0.0964, 'CI': (0.0788, 0.11), 'mean': 0.09623999999999999, 'std': 0.013116340953177453},
    'Means': {'p_C': 0.12116137064813201, 'CI': (0.10567834935848158, 0.13408197987526282), 'mean': 0.12062971258005899, 'std': 0.010329231195737199},
    'EMD': {'p_C': 0.12156986188389857, 'CI': (0.10695540862358027, 0.13506864157907), 'mean': 0.12130475132716852, 'std': 0.010077862989318908},
    'KDE': {'p_C': 0.11653533406952703, 'CI': (0.10008488391610758, 0.13113053107870062), 'mean': 0.11330220847636982, 'std': 0.011794233996819635}
    }

expected_n_boot_10_n_mix_10 = {
    'Excess': {'p_C': 0.0964, 'CI': (0.0388, 0.1116), 'mean': 0.071952, 'std': 0.020117039941303493},
    'Means': {'p_C': 0.12116137064813201, 'CI': (0.08675545060213877, 0.15875832661988384), 'mean': 0.1248328111839013, 'std': 0.019118462180588044},
    'EMD': {'p_C': 0.12156986188389857, 'CI': (0.07862428927064141, 0.14534424541114516), 'mean': 0.11489674057535156, 'std': 0.018446112762141968},
    'KDE': {'p_C': 0.11653533406952703, 'CI': (0.06956800324117474, 0.1446109574892474), 'mean': 0.11398743016947095, 'std': 0.01926771502334951}
    }


@pytest.mark.parametrize("n_boot,n_mix,seed,nproc,expected", [
    (0, 0, seed, 1, expected_point),
    (0, 10, seed, 1, expected_point),  # NOTE: Mixture modelling is skipped if n_boot==0
    (10, 0, seed, 1, expected_n_boot_10),
    (10, 10, seed, 1, expected_n_boot_10_n_mix_10),
    (10, 0, seed, nproc, expected_n_boot_10_parallel),  # NOTE: These results differ from the serial runs but depend on n_boot not n_jobs
    (10, 10, seed, nproc, expected_n_boot_10_n_mix_10),
])
def test_anaylse_mixture_parameterised(scores_pC010, n_boot, n_mix, seed, nproc, expected):
    summary, bootstraps = dpe.analyse_mixture(scores_pC010, n_boot=n_boot, n_mix=n_mix, seed=seed, n_jobs=nproc)
    for method, results in summary.items():
        for metric, value in results.items():
            assert np.all(np.isclose(value, expected[method][metric])), \
            f"{value} != expected[{method}][{metric}]={expected[method][metric]}"


# def test_anaylse_mixture(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=0, n_mix=0, seed=seed, true_pC=p_C)
#     summary, bootstraps = results

#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)


# def test_anaylse_mixture_mix(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=0, n_mix=10, seed=seed, true_pC=p_C)
#     summary, bootstraps = results

#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)


# def test_anaylse_mixture_boot(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=10, n_mix=0, seed=seed, true_pC=p_C)
#     summary, bootstraps = results

#     # Proportion estimates
#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

#     # Bootstrap estimates
#     assert np.isclose(summary["EMD"]["mean"], 0.12269521263948222), summary["EMD"]["mean"]
#     assert np.isclose(summary["EMD"]["std"], 0.013433323691734361), summary["EMD"]["std"]
#     assert np.isclose(summary["EMD"]["CI"][0], 0.11028732209621961), summary["EMD"]["CI"][0]
#     assert np.isclose(summary["EMD"]["CI"][1], 0.14805938902974064), summary["EMD"]["CI"][1]

#     assert np.isclose(summary["Excess"]["mean"], 0.10216)
#     assert np.isclose(summary["Excess"]["std"], 0.01102027222894244)
#     assert np.isclose(summary["Excess"]["CI"][0], 0.0952)
#     assert np.isclose(summary["Excess"]["CI"][1], 0.1248)

#     assert np.isclose(summary["KDE"]["mean"], 0.12278166955266238)
#     assert np.isclose(summary["KDE"]["std"], 0.012485646023580961)
#     assert np.isclose(summary["KDE"]["CI"][0], 0.10882118268399207)
#     assert np.isclose(summary["KDE"]["CI"][1], 0.1448788902735669)

#     assert np.isclose(summary["Means"]["mean"], 0.12279898550753088)
#     assert np.isclose(summary["Means"]["std"], 0.01336121417716939)
#     assert np.isclose(summary["Means"]["CI"][0], 0.11085840202841786)
#     assert np.isclose(summary["Means"]["CI"][1], 0.14815690928847491)


# def test_anaylse_mixture_boot_parallel(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=10, n_mix=0, seed=seed, n_jobs=nproc, true_pC=p_C)
#     summary, bootstraps = results

#     # Proportion estimates
#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

#     # Bootstrap estimates
#     # NOTE: These are slightly different than when n_jobs==1
#     assert np.isclose(summary["EMD"]["mean"], 0.12267561935887468), summary["EMD"]["mean"]
#     assert np.isclose(summary["EMD"]["std"], 0.00860512889035671), summary["EMD"]["std"]
#     assert np.isclose(summary["EMD"]["CI"][0], 0.10911018704388409), summary["EMD"]["CI"][0]
#     assert np.isclose(summary["EMD"]["CI"][1], 0.13124582773285004), summary["EMD"]["CI"][1]

#     assert np.isclose(summary["Excess"]["mean"], 0.09759999999999999), summary["Excess"]["mean"]
#     assert np.isclose(summary["Excess"]["std"], 0.0100860299424501), summary["Excess"]["std"]
#     assert np.isclose(summary["Excess"]["CI"][0], 0.0776), summary["Excess"]["CI"][0]
#     assert np.isclose(summary["Excess"]["CI"][1], 0.1108), summary["Excess"]["CI"][1]

#     assert np.isclose(summary["KDE"]["mean"], 0.11523613740188092), summary["KDE"]["mean"]
#     assert np.isclose(summary["KDE"]["std"], 0.011402856318140971), summary["KDE"]["std"]
#     assert np.isclose(summary["KDE"]["CI"][0], 0.09797229341051193), summary["KDE"]["CI"][0]
#     assert np.isclose(summary["KDE"]["CI"][1], 0.13136957654751658), summary["KDE"]["CI"][1]

#     assert np.isclose(summary["Means"]["mean"], 0.12283908114071776), summary["Means"]["mean"]
#     assert np.isclose(summary["Means"]["std"], 0.008544651916214157), summary["Means"]["std"]
#     assert np.isclose(summary["Means"]["CI"][0], 0.11014270881728092), summary["Means"]["CI"][0]
#     assert np.isclose(summary["Means"]["CI"][1], 0.13095145348487366), summary["Means"]["CI"][1]


# def test_anaylse_mixture_mix_boot(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=10, n_mix=10, seed=seed, true_pC=p_C)
#     summary, bootstraps = results

#     # Proportion estimates
#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

#     # Bootstrap estimates
#     assert np.isclose(summary["EMD"]["mean"], 0.11991567686380979)
#     assert np.isclose(summary["EMD"]["std"], 0.01879614372677909)
#     assert np.isclose(summary["EMD"]["CI"][0], 0.0835170826819473)
#     assert np.isclose(summary["EMD"]["CI"][1], 0.15547233358398993)

#     assert np.isclose(summary["Excess"]["mean"], 0.08225199999999999)
#     assert np.isclose(summary["Excess"]["std"], 0.016321240639118093)
#     assert np.isclose(summary["Excess"]["CI"][0], 0.0532)
#     assert np.isclose(summary["Excess"]["CI"][1], 0.1176)

#     assert np.isclose(summary["KDE"]["mean"], 0.1180030865344234)
#     assert np.isclose(summary["KDE"]["std"], 0.017439527912894638)
#     assert np.isclose(summary["KDE"]["CI"][0], 0.09177886669333177)
#     assert np.isclose(summary["KDE"]["CI"][1], 0.15453785682429844)

#     assert np.isclose(summary["Means"]["mean"], 0.10913808007686349)
#     assert np.isclose(summary["Means"]["std"], 0.018587226882665266)
#     assert np.isclose(summary["Means"]["CI"][0], 0.0719092323672722)
#     assert np.isclose(summary["Means"]["CI"][1], 0.1400325155522334)


# def test_anaylse_mixture_mix_boot_parallel(scores_pC010):
#     results = dpe.analyse_mixture(scores_pC010, n_boot=10, n_mix=10, seed=seed, n_jobs=nproc, true_pC=p_C)
#     summary, bootstraps = results

#     # Proportion estimates
#     assert np.isclose(summary["EMD"]["p_C"], 0.12156986188389857)
#     assert np.isclose(summary["Excess"]["p_C"], 0.0964)
#     assert np.isclose(summary["KDE"]["p_C"], 0.11653533406952703)
#     assert np.isclose(summary["Means"]["p_C"], 0.12116137064813201)

#     # Bootstrap estimates
#     assert np.isclose(summary["EMD"]["mean"], 0.11991567686380979)
#     assert np.isclose(summary["EMD"]["std"], 0.01879614372677909)
#     assert np.isclose(summary["EMD"]["CI"][0], 0.0835170826819473)
#     assert np.isclose(summary["EMD"]["CI"][1], 0.15547233358398993)

#     assert np.isclose(summary["Excess"]["mean"], 0.08225199999999999)
#     assert np.isclose(summary["Excess"]["std"], 0.016321240639118093)
#     assert np.isclose(summary["Excess"]["CI"][0], 0.0532)
#     assert np.isclose(summary["Excess"]["CI"][1], 0.1176)

#     assert np.isclose(summary["KDE"]["mean"], 0.1180030865344234)
#     assert np.isclose(summary["KDE"]["std"], 0.017439527912894638)
#     assert np.isclose(summary["KDE"]["CI"][0], 0.09177886669333177)
#     assert np.isclose(summary["KDE"]["CI"][1], 0.15453785682429844)

#     assert np.isclose(summary["Means"]["mean"], 0.10913808007686349)
#     assert np.isclose(summary["Means"]["std"], 0.018587226882665266)
#     assert np.isclose(summary["Means"]["CI"][0], 0.0719092323672722)
#     assert np.isclose(summary["Means"]["CI"][1], 0.1400325155522334)
