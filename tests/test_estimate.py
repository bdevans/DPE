import sys

import numpy as np

print(sys.path)

import dpe.datasets as ds
import dpe


seed = 0
np.random.seed(seed)

p_C = 0.1
scores = ds.generate_dataset(p_C=p_C)

def test_anaylse_mixture():
    results = dpe.analyse_mixture(scores, n_boot=0, n_mix=0, true_pC=p_C)
    summary, bootstraps = results

    assert np.isclose(summary["EMD"]["p_C"], 0.09405079822913687)
    assert np.isclose(summary["Excess"]["p_C"], 0.062)
    assert np.isclose(summary["KDE"]["p_C"], 0.09287585631089693)
    assert np.isclose(summary["Means"]["p_C"], 0.09357232105246638)
