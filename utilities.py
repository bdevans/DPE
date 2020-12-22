import time

import numpy as np
from sklearn.neighbors import KernelDensity


class Timer:
    """
    Class for timing blocks of code.
    Adapteed from http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
    
    Examples
    --------
    >>> with Timer() as t:
    >>>    run code...
    Execution took <t>s)
    """
    # interval = 0
    def __init__(self, clock=time.perf_counter):
        """clock: 
            time.perf_counter (wall time)
            time.process_time (processor time)
        """
        self.start = 0
        self.end = 0
        self.interval = 0
        self.clock = clock

    def __enter__(self):
        self.start = self.clock()
        return self

    def __exit__(self, *args):
        self.end = self.clock()
        self.interval = self.end - self.start
        print(f"{self.interval:.3g}s")

    def __str__(self):
        return f"{self.interval:.3g}s"

    def reset(self):
        """Reset timer to 0."""
        self.start = 0
        self.end = 0
        self.interval = 0


def get_fpr_tpr(scores, bins):

    # scores, bins
    # method = 'fd'
    # all_refs = [*scores["R_C"], *scores["R_N"]]
    # # probas_, edges_r = np.histogram(all_refs, bins=method, range=(bins["min"], bins["max"]), density=True)
    # probas_, edges_a = np.histogram(all_refs, bins=bins["edges"], density=True)

    # TPR = TP / P
    # FPR = FP / N
    # x=FPR, y=TPR
    # p = 1. * np.arange(len(all_refs)) / (len(all_refs) - 1)
    hist_1, _ = np.histogram(scores["R_C"], bins=bins["edges"], density=False)
    hist_2, _ = np.histogram(scores["R_N"], bins=bins["edges"], density=False)
    if scores["R_C"].mean() > scores["R_N"].mean():
        hist_p = hist_1
        hist_n = hist_2
    else:
        hist_p = hist_2
        hist_n = hist_1

    cum_p = np.cumsum(hist_p)  # R_C := cases := positive
    cond_P = hist_p.sum()
    tp = cond_P - cum_p  # Must subtract from P since cumsum grows the opposite way
    
    cum_n = np.cumsum(hist_n)  # R_N := non-cases := negative
    cond_N = hist_n.sum()
    tn = cum_n

    # if scores["R_C"].mean() > scores["R_N"].mean():
    #     tp = np.flip(tp)
    #     tn = np.flip(tn)

    # Assume mean(GRS_c) > mean(GRS_n)
    tpr = tp / cond_P
    fpr = 1 - (tn / cond_N)

    # fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
    # fpr = np.r_[0, fpr]
    # tpr = np.r_[0, tpr]
    return fpr, tpr


def fit_kernels(scores, bw, kernel='gaussian'):
    """No longer used."""
    kernels = {}
    for label, data in scores.items():
        X = data[:, np.newaxis]
        kernels[label] = KernelDensity(kernel=kernel, bandwidth=bw,
                                       atol=0, rtol=1e-4).fit(X)
    return kernels
