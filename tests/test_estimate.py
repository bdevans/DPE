import os

import pytest
import numpy as np

import dpe.datasets as ds
import dpe


seed = 0
# p_C = 0.1
nproc = os.cpu_count()


@pytest.fixture
def scores_pC010_size100():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=100, seed=seed)

expected_point_size100 = {
    'Excess': {'p_C': 0.14}, 
    'Means': {'p_C': 0.18293067873890895}, 
    'EMD': {'p_C': 0.1598270527703443}, 
    'KDE': {'p_C': 0.22486535485520434}
    }

expected_size100_n_boot10 = {
    'Excess': {'p_C': 0.14, 'CI': (0.0, 0.28), 'mean': 0.17600000000000002, 'std': 0.09583318840568752, 'bias': 0.0}, 
    'Means': {'p_C': 0.18293067873890895, 'CI': (0.11636027513235522, 0.33880589896255464), 'mean': 0.19488787944082625, 'std': 0.10328422616643053, 'bias': -0.008438343021288042}, 
    'EMD': {'p_C': 0.1598270527703443, 'CI': (0.02119497381956048, 0.2883579618350394), 'mean': 0.17272677210846873, 'std': 0.08456747077157699, 'bias': 0.014759182335856796}, 
    'KDE': {'p_C': 0.22486535485520434, 'CI': (0.029069533605505137, 0.3669878169136295), 'mean': 0.24316495927493706, 'std': 0.10688796970498039, 'bias': 0.02543108175970124}
    }

expected_size100_n_boot10_n_mix10 = {
    'Excess': {'p_C': 0.14, 'CI': (0.0, 0.34), 'mean': 0.1502, 'std': 0.10250834112402755, 'bias': 0.0}, 
    'Means': {'p_C': 0.18293067873890895, 'CI': (0.0, 0.4686396758186981), 'mean': 0.16170378791626627, 'std': 0.11415563390587638, 'bias': -0.013712238845887764}, 
    'EMD': {'p_C': 0.1598270527703443, 'CI': (0.018079589403520724, 0.4289956832203804), 'mean': 0.18648509022269402, 'std': 0.11646875624771075, 'bias': 0.010452929491985036}, 
    'KDE': {'p_C': 0.22486535485520434, 'CI': (1.2692213373587228e-09, 0.52946066246552), 'mean': 0.2371214922403585, 'std': 0.15130456753787117, 'bias': -0.00023046799646284089}
    }

expected_size100_n_boot10_parallel = {
    'Excess': {'p_C': 0.14, 'CI': (0.0, 0.28), 'mean': 0.17600000000000002, 'std': 0.09583318840568752, 'bias': 0.0}, 
    'Means': {'p_C': 0.18293067873890895, 'CI': (0.11636027513235522, 0.33880589896255464), 'mean': 0.19488787944082625, 'std': 0.10328422616643053, 'bias': -0.008438343021288042}, 
    'EMD': {'p_C': 0.1598270527703443, 'CI': (0.02119497381956048, 0.2883579618350394), 'mean': 0.17272677210846873, 'std': 0.08456747077157699, 'bias': 0.014759182335856796}, 
    'KDE': {'p_C': 0.22486535485520434, 'CI': (0.029069533605505137, 0.3669878169136295), 'mean': 0.24316495927493706, 'std': 0.10688796970498039, 'bias': 0.02543108175970124}
    }

# @pytest.mark.fast
@pytest.mark.parametrize("n_boot,n_mix,seed,nproc,expected", [
    (0, 0, seed, 1, expected_point_size100),
    (0, 10, seed, 1, expected_point_size100),  # NOTE: Mixture modelling is skipped if n_boot==0
    (10, 0, seed, 1, expected_size100_n_boot10),
    (10, 10, seed, 1, expected_size100_n_boot10_n_mix10),
    (10, 10, seed, nproc, expected_size100_n_boot10_n_mix10),
    (10, 0, seed, nproc, expected_size100_n_boot10_parallel),  # NOTE: These results differ from the serial runs but depend on n_boot not n_jobs
])
def test_anaylse_mixture_parameterised(scores_pC010_size100, n_boot, n_mix, seed, nproc, expected):
    summary, bootstraps = dpe.analyse_mixture(scores_pC010_size100, n_boot=n_boot, n_mix=n_mix, seed=seed, n_jobs=nproc)
    for method, results in summary.items():
        for metric, value in results.items():
            assert np.all(np.isclose(value, expected[method][metric])), \
            f"{value} != expected[{method}][{metric}]={expected[method][metric]}"


@pytest.fixture
def scores_pC010_size5000():
    """Generate synthetic data with p_C=0.1"""
    return ds.generate_dataset(p_C=0.1, size=5000, seed=seed)


expected_point_size5000 = {
    'Excess': {'p_C': 0.0964},
    'Means': {'p_C': 0.12116137064813201},
    'EMD': {'p_C': 0.12156986188389857},
    'KDE': {'p_C': 0.11653533406952703}
    }

expected_size5000_n_boot10 = {

    }

expected_size5000_n_boot10_n_mix10 = {

    }

expected_size5000_n_boot10_parallel = {

    }


# @pytest.mark.skip
# # @pytest.mark.slow
# @pytest.mark.parametrize("n_boot,n_mix,seed,nproc,expected", [
#     (0, 0, seed, 1, expected_point_size5000),
#     (0, 10, seed, 1, expected_point_size5000),  # NOTE: Mixture modelling is skipped if n_boot==0
#     (10, 0, seed, 1, expected_size5000_n_boot10),
#     (10, 10, seed, 1, expected_size5000_n_boot10_n_mix10),
#     (10, 10, seed, nproc, expected_size5000_n_boot10_n_mix10),
#     (10, 0, seed, nproc, expected_size5000_n_boot10_parallel),  # NOTE: These results differ from the serial runs but depend on n_boot not n_jobs
# ])
# def test_anaylse_mixture_parameterised(scores_pC010_size5000, n_boot, n_mix, seed, nproc, expected):
#     summary, bootstraps = dpe.analyse_mixture(scores_pC010_size5000, n_boot=n_boot, n_mix=n_mix, seed=seed, n_jobs=nproc)
#     for method, results in summary.items():
#         for metric, value in results.items():
#             assert np.all(np.isclose(value, expected[method][metric])), \
#             f"{value} != expected[{method}][{metric}]={expected[method][metric]}"


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
