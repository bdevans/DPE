import os
import time

import numpy as np
from sklearn.neighbors import KernelDensity

# import dpe
# from dpe.estimate import calc_conf_intervals
from . config import _ALL_METHODS_


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


def format_seconds(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m,   60)
    return u'%d:%02d:%02d' % (h, m, s)


# Let's use FD!
def estimate_bins(data, bin_range=None, verbose=0):
    """Generate GRS bins through data-driven methods in `np.histogram`.

    These methods include:
    ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']

    For more information see:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
    """
    # TODO: Refactor to pass one method and return only that dictionary

    # 'scott': n**(-1./(d+4))
    # kdeplot also uses 'silverman' as used by scipy.stats.gaussian_kde
    # (n * (d + 2) / 4.)**(-1. / (d + 4))
    # with n the number of data points and d the number of dimensions
    line_width = 49

    hist = {}
    bins = {}
    if verbose:
        print("  Method | Data |  n  |  width  |      range     ", flush=True)
        print("=" * line_width)
    for method in ['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']:
        all_scores = []
        all_refs = []
        for group, scores in data.items():
            all_scores.extend(scores)
            if group != "Mix":
                all_refs.extend(scores)
            # else:  # Add extremes to ensure the full range is spanned
            #     all_refs.extend([min(scores), max(scores)])
            if bin_range is None:
                bin_range = (min(all_scores), max(all_scores))
            if verbose > 1:
                _, bin_edges = np.histogram(scores, bins=method, range=bin_range)
                print(f" {method:>7} | {group:>4} | {len(bin_edges)-1:>3} | {bin_edges[1]-bin_edges[0]:<7.5f} | [{bin_edges[0]:5.3}, {bin_edges[-1]:5.3}]")
                # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format(group, method, bin_edges[1]-bin_edges[0], len(bin_edges)-1, bin_edges[0], bin_edges[-1]))

        h_r, edges_r = np.histogram(all_refs, bins=method,
                                    range=(min(all_scores), max(all_scores)))
        if verbose > 1:
            print("-" * line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "Refs", len(edges_r)-1, edges_r[1]-edges_r[0], edges_r[0], edges_r[-1]))

        h_a, edges_a = np.histogram(all_scores, bins=method, range=bin_range)  # Return edges

        if verbose:
            if verbose > 1:
                print("-" * line_width)
            print(" {:>7} | {:>4} | {:>3} | {:<7.5f} | [{:5.3}, {:5.3}]"
                  .format(method, "All", len(edges_a)-1, edges_a[1]-edges_a[0], edges_a[0], edges_a[-1]))
            # print("{:4} {:>7}: width = {:<7.5f}, n_bins = {:>4,}, range = [{:5.3}, {:5.3}]".format("All", method, b['width'], b['n'], b['min'], b['max']))
            if verbose > 1:
                print("=" * line_width)
            else:
                print("-" * line_width)

        # h, edges = h_a, edges_a
        h, edges = h_r, edges_r
        hist[method] = h
        bins[method] = {'width': edges[1] - edges[0],
                        'min': edges[0],
                        'max': edges[-1],
                        'edges': edges,
                        'centers': (edges[:-1] + edges[1:]) / 2,
                        'n': len(edges) - 1}
    return hist, bins


def get_fpr_tpr(scores, bins):

    # scores, bins
    # method = 'fd'
    # all_refs = [*scores["R_C"], *scores["R_N"]]
    # # probas_, edges_r = np.histogram(all_refs, bins=method, range=(bins["min"], bins["max"]), density=True)
    # probas_, edges_a = np.histogram(all_refs, bins=bins["edges"], density=True)

    # TPR = TP / P
    # FPR = FP / N
    # x=FPR, y=TPR
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
    tp = cond_P - cum_p  # Must subtract from P since cumsum grows the opposite way

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


def construct_mixture(R_C, R_N, p_C, size):
    assert(0.0 <= p_C <= 1.0)
    n_C = int(round(size * p_C))
    n_N = size - n_C

    # Construct mixture
    mixture = np.concatenate((np.random.choice(R_C, n_C, replace=True),
                              np.random.choice(R_N, n_N, replace=True)))
    return mixture


def fit_kernels(scores, bw, kernel='gaussian'):
    """No longer used."""
    kernels = {}
    for label, data in scores.items():
        X = data[:, np.newaxis]
        kernels[label] = KernelDensity(kernel=kernel, bandwidth=bw,
                                       atol=0, rtol=1e-4).fit(X)
    return kernels


def load_accuracy(data_dir, label):

    proportions = np.load(os.path.join(data_dir, f"proportions_{label}.npy"))
    sample_sizes = np.load(os.path.join(data_dir, f"sample_sizes_{label}.npy"))
    # PROPORTIONS_R_N = PROPORTIONS_R_C[::-1]

    # Dictionary of p1 errors
    point_estimates = {}
    boots_estimates = {}

    for method in _ALL_METHODS_:
        point_file = os.path.join(data_dir, f"point_{method}_{label}.npy")
        boots_file = os.path.join(data_dir, f"boots_{method}_{label}.npy")
        if os.path.isfile(point_file):
            point_estimates[method] = np.load(point_file)
        if os.path.isfile(boots_file):
            boots_estimates[method] = np.load(boots_file)

    return point_estimates, boots_estimates, proportions, sample_sizes
