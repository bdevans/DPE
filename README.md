Distribution Proportion Estimation
==================================

Methods for estimating the prevalence of cases in a mixture population based on genetic risk scores.

[![GitHub license](https://img.shields.io/github/license/bdevans/DPE)](https://github.com/bdevans/DPE/blob/master/LICENSE.txt)
[**DOI: 10.1101/2020.02.20.20025528**](https://doi.org/10.1101/2020.02.20.20025528)

------------------------------------------------------------------------

This repository contains the Python 3 implementation of the proportion estimation algorithms, which was developed and run with Python 3.7.6. The following instructions assume that you have a working Python (>=3.6) installation obtained either through Anaconda (recommended) or through other means (which requires pip). 

Installation
------------

0. Install Python >= 3.6. 
1. Add this folder to your `PYTHONPATH`.
2. Install the requirements using one of the provided requirements files:
   1. If you use a standard Python distribution: `pip3 install -r requirements.txt`.
   2. Or, if you use Anaconda: `conda env create -f environment.yml`. Then run `source activate dpe` to activate the environment. 

This should take approximately 1-2 minutes to complete, depending on the speed of your internet connection and the number of dependencies already satisfied. 

The examples given were tested with Python 3.7.6 on macOS 10.14.6 (18G3020) running in a Docker container (version 19.03.5, build 633a0ea). The exact packages installed with `pip` during testing are given in `frozen_dependencies.txt`. 

Running the worked examples
---------------------------

Three main code files are relevant for review purposes:
1. `scripts/run_examples.py` the main script for applying the proportion estimation algorithms to the example data sets.
2. `dpe/datasets.py` has utilities for generating, loading and saving synthetic data sets.
3. `dpe/estimate.py` has the main routines for estimating proportions in a mixture distribution.

Once the requirements are installed (and the environment activated if necessary) run the example script with:

```
python scripts/run_examples.py
```

The proportion estimates and confidence intervals are then generated, with a plain text summary printed to the console (when verbose > 0) and written to a log file (named after the data set file with a `.log` extension). 

The analysis parameters may be changed by editing the `run_examples.py` script. It is recommended to keep the total number of mixture bootstraps (`n_mix * n_boot`) below around 10,000 when using the KDE method, as this may taken a long time to finish. 

Optionally, the file `datasets.py` may be edited to change the construction parameters and generate new data sets to analyse. 

### Expected Output

When running with the parameters given in the manuscript (N_M = 100 N_B = 1000) on a 2015 15" Macbook Pro (2.8 GHz Intel Core i7) with 8 threads, processing each data set takes around 1h15m. The majority of this time is spent running the KDE algorithm (the other three each take approximately 1m each). This run time can be reduced considerably by reducing the number of mixtures generated (`n_mix`) and/or the number of bootstraps generated for each mixture (`n_boot`). Accordingly, the simulation time has been reduced in `run_examples.py` for demonstration purposes by reducing the number of bootstraps (N_B = 100) such that the run time is approximately 10m per data set (although this can be edited for longer runs). The output produced for the first data set is given below for N_B = 100:

```
================================================================================
Running on example dataset: p_C = 0.25
Loading dataset: example_pC025...
Running 100 bootstraps with 8 processors...
Method: 100%|█████████████████████████████████████| 4/4 [09:07<00:00, 136.98s/it]

    Method    |   Estimated pC    |   Estimated pN    
======================================================
 Excess point | 0.18200           | 0.81800           
 Excess (µ±σ) | 0.13694 +/- 0.019 | 0.86306 +/- 0.019 
 C.I. (95.0%) | 0.18920 , 0.26361 | 0.73639 , 0.81080 
 Corrected    | 0.22706           | 0.77294           
------------------------------------------------------
 Means  point | 0.24279           | 0.75721           
 Means  (µ±σ) | 0.24490 +/- 0.018 | 0.75510 +/- 0.018 
 C.I. (95.0%) | 0.20469 , 0.27637 | 0.72363 , 0.79531 
 Corrected    | 0.24069           | 0.75931           
------------------------------------------------------
 EMD    point | 0.24313           | 0.75687           
 EMD    (µ±σ) | 0.24357 +/- 0.019 | 0.75643 +/- 0.019 
 C.I. (95.0%) | 0.20500 , 0.27984 | 0.72016 , 0.79500 
 Corrected    | 0.24269           | 0.75731           
------------------------------------------------------
 KDE    point | 0.24843           | 0.75157           
 KDE    (µ±σ) | 0.24595 +/- 0.022 | 0.75405 +/- 0.022 
 C.I. (95.0%) | 0.20809 , 0.29445 | 0.70555 , 0.79191 
 Corrected    | 0.25090           | 0.74910           
------------------------------------------------------
 Ground Truth | 0.25000           | 0.75000           
======================================================

Elapsed time = 548.460 seconds
================================================================================ 

```

The longer run time output from processing the `example_pC025` data set with N_M = 100 and N_B = 1000 is given below:

```
================================================================================
Running on example dataset: p_C = 0.25
Loading dataset: example_pC025...
Running 1000 bootstraps with 8 processors...
Method: 100%|█████████████████████████████████████| 4/4 [1:14:08<00:00, 1112.13s/it]

    Method    |   Estimated pC    |   Estimated pN    
======================================================
 Excess point | 0.18200           | 0.81800           
 Excess (µ±σ) | 0.13752 +/- 0.019 | 0.86248 +/- 0.019 
 C.I. (95.0%) | 0.18920 , 0.26200 | 0.73800 , 0.81080 
 Corrected    | 0.22648           | 0.77352           
------------------------------------------------------
 Means  point | 0.24279           | 0.75721           
 Means  (µ±σ) | 0.24466 +/- 0.018 | 0.75534 +/- 0.018 
 C.I. (95.0%) | 0.20485 , 0.27659 | 0.72341 , 0.79515 
 Corrected    | 0.24093           | 0.75907           
------------------------------------------------------
 EMD    point | 0.24313           | 0.75687           
 EMD    (µ±σ) | 0.24374 +/- 0.019 | 0.75626 +/- 0.019 
 C.I. (95.0%) | 0.20468 , 0.27912 | 0.72088 , 0.79532 
 Corrected    | 0.24252           | 0.75748           
------------------------------------------------------
 KDE    point | 0.24843           | 0.75157           
 KDE    (µ±σ) | 0.24610 +/- 0.021 | 0.75390 +/- 0.021 
 C.I. (95.0%) | 0.20822 , 0.29159 | 0.70841 , 0.79178 
 Corrected    | 0.25075           | 0.74925           
------------------------------------------------------
 Ground Truth | 0.25000           | 0.75000           
======================================================

Elapsed time = 4450.042 seconds
================================================================================ 

```

Additionally a `results` directory will be created with a subdirectory for each data set processed containing a `csv` file with the initial point estimates and bootstrap estimates for each method. 

Reproducibility
---------------

The results are reproducible by default since a seed is set (42) for the pseudo random number generator. This seed may be changed (or set to `None` for a random seed) or set to any integer in the range `[0, 2^32)` to explore variations in results due to stochasticity in sampling. 

Running on your data
--------------------

The main requirement is to prepare a dictionary (`dict`) containing the keys `R_C`, `R_N` and `Mix`. The associated values should be (one dimensional) arrays (or lists) of the GRS scores for the "Cases Reference" (`R_C`) distribution, the "Non-cases Reference" (`R_N`) distribution and the "Mixture" distribution (`Mix`) respectively. 

Alternatively a `csv` file may be prepared and loaded with the function `datasets.load_dataset(filename)` as demonstrated in the `run_examples.py` script. The `csv` file should contain the header `Group,GRS` followed by pairs of `code,score` values (one per line for each GRS score) where code is `1` for the Cases Reference distribution, `2` for the Non-cases Reference distribution and `3` for the Mixture distribution. 

Once the GRS scores have been prepared in a suitable form, they may be passed to the `analyse_mixture()` function as demonstrated in the `run_examples.py` script. Further details about this function are given in the next section. 

Pseudocode
----------

The main algorithm is as follows. Details of the specific methods may be found in the methods section of the accompanying manuscript. 

```sh
bins <-- generate_bins({R_C, R_N, Mix}, method='fd')  # Freedman-Diaconis
p^hat <-- {}  # Dictionary (hashtable) of proportion estimates
p^cor <-- {}  # Dictionary (hashtable) of corrected proportion estimates
p^mbe <-- {}  # Dictionary (hashtable) of mixture-bootstrap estimates
CI <-- {}  # Dictionary (hashtable) of confidence intervals (2-tuple)
for meth in ["Excess", "Means", "EMD", "KDE"]:
    p^hat[meth] <-- get_point_estimate({R_C, R_N, Mix}, bins, meth)  # Proportion of cases
    # Calculate confidence intervals around the initial proportion estimates
    p^mbe[meth] = []  # Create empty list of estimates for each method
    for m in 1 to N_M:  # Number of Monte Carlo mixtures
        Mix_meth_m <-- get_monte_carlo_mixture({R_C, R_N}, p^hat[meth])
        for b in 1 to N_B:  # Number of bootstraps
            Mix_meth_m_b <-- get_bootstrap({Mix_meth_m}, bins)  # Sample with replacement
            p_meth_m_b <-- get_point_estimate({R_C, R_N, Mix_meth_m_b}, bins, meth)
            p^mbe[meth].append(p_meth_m_b)
        end
    end
    p^cor[meth] <-- correct_estimate(p^hat[meth], p^mbe[meth])  # Use the set of all bootstraps of all mixtures for each method
    CI[meth] <-- get_confidence_intervals(p^cor[meth], p^mbe[meth], alpha=0.05)
end
```

Explanation of the main function
--------------------------------

```python
def analyse_mixture(scores, bins='fd', methods='all',
                    n_boot=1000, boot_size=-1, n_mix=0, alpha=0.05,
                    ci_method="bca", correct_bias=False, seed=None,
                    n_jobs=1, verbose=1, true_pC=None, logfile=''):
```

### Inputs

- `scores` (`dict`): A required dictionary of the form, `{'R_C': array_of_cases_scores, 'R_N': array_of_non-cases_scores, 'Mix': array_of_mixture_scores}`.
- `bins` (`str`): A string specifying the binning method: `['auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt']`. Default: `'fd'`. Alternatively, a dictionary, `{'width': bin_width, 'min', min_edge, 'max': max_edge, 'edges': array_of_bin_edges, 'centers': array_of_bin_centers, 'n': number_of_bins}`.
- `methods` (`str`): A string with the name of the method or `'all'` to run all methods (default). Alternatively, a list of method names (strings), `["Excess", "Means", "EMD", "KDE"]`, or a dictionary of (bool) flags, `{'Excess': True, 'Means': True, 'EMD': True, 'KDE': True}`.
- `n_boot` (`int`): Number of bootstraps of the mixture to generate. Default: `1000`.
- `boot_size` (`int`): The size of each mixture bootstrap. Default is the same size as the mixture.
- `n_mix` (`int`): Number of mixtures to construct based on the initial point estimate. Default: `0`.
- `alpha` (`float`): The alpha value for calculating confidence intervals from bootstrap distributions. Default: `0.05`.
- `ci_method` (`str`): The name of the method used to calculate the confidence intervals Default: `bca`.
- `correct_bias` (`bool`): A boolean flag specifing whether to apply the bootstrap correction method or not. Default: `False`.
- `seed` (`int`): An optional value to seed the random number generator with (in the range [0, (2^32)-1]) for reproducibility of sampling used for confidence intervals. Defaults: `None`.
- `n_jobs` (`int`): Number of bootstrap jobs to run in parallel. Default: `1`. (`n_jobs = -1` runs on all CPUs).
- `verbose` (`int`): Integer to control the level of output (`0`, `1`, `2`). Set to `-1` to turn off all console output except the progress bars.
- `true_pC` (`float`): Optionally pass the true proportion for showing the comparison with estimated proportion(s).
- `logfile` (`str`): Optional filename for the output logs. Default: `"proportion_estimates.log"`.
 
### Outputs

(summary, bootstraps) (`tuple`): A tuple consisting of the following data structures.

- summary (`dict`): A nested dictionary with a key for each estimation method within which is a dictionary with the following keys:
    - `p_C` : the prevalence estimate
    
    Optionally, if bootstrapping is used:
    - `CI` : the confidence intervals around the prevalence
    - `mean` : the mean of the bootstrapped estimates
    - `std` : the standard deviation of the bootstrap estimates
    - `p_cor_C` : the correct prevalence estimate when `correct_bias == True`
- bootstraps (`DataFrame`): A `pandas` dataframe of the proportion estimates. The first row is the point estimate. The remaining `n_boot * n_mix` rows are the bootstrapped estimates. Each column is the name of the estimation method.

Additionally the logfile is written to the working directory.


Methods
-------

We bootstrap (i.e. sample with replacement) from the available data, systematically varying sample size and mixture proportion. We then apply the following methods to yield estimates of the true proportion and from those, calculate the errors throughout the bootstrap parameter space.

These methods assume that higher GRS is associated with cases. 

1. Means

    ```python
    p_hat_C = (RM.mean() - mu_N) / (mu_C - mu_N)
    p_hat_C = np.clip(p_hat_C, 0.0, 1.0)
    ```

2. [Excess](https://www.thelancet.com/journals/landia/article/PIIS2213-8587(17)30362-5/fulltext)

    ```python
    number_low = len(RM[RM <= population_median])
    number_high = len(RM[RM > population_median])
    p_hat_C = (number_high - number_low) / len(RM)
    p_hat_C = np.clip(p_hat_C, 0.0, 1.0)
    ```

3. [EMD](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)
    ```math
    EMD(P,Q)={\frac {\sum _{i=1}^{m}\sum _{j=1}^{n}f_{i,j}d_{i,j}}{\sum _{i=1}^{m}\sum _{j=1}^{n}f_{i,j}}}
    ```

    ```python
    def interpolate_CDF(scores, x_i, min_edge, max_edge):
        """Interpolate the cumulative density function of the scores at the points in the array `x_i`.
        """

        x = [min_edge, *sorted(scores), max_edge]
        y = np.linspace(0, 1, num=len(x), endpoint=True)
        (iv, ii) = np.unique(x, return_index=True)
        return np.interp(x_i, iv, y[ii])

    # Interpolate the cdfs at the same points for comparison
    CDF_1 = interpolate_CDF(scores["R_C"], bins['centers'], bins['min'], bins['max'])
    CDF_2 = interpolate_CDF(scores["R_N"], bins['centers'], bins['min'], bins['max'])
    CDF_M = interpolate_CDF(RM, bins['centers'], bins['min'], bins['max'])
    
    # EMDs computed with interpolated CDFs
    EMD_1_2 = sum(abs(CDF_1 - CDF_2))
    EMD_M_1 = sum(abs(CDF_M - CDF_1))
    EMD_M_2 = sum(abs(CDF_M - CDF_2))

    p_hat_C = 0.5 * (1 + (EMD_M_2 - EMD_M_1) / EMD_1_2)
    ```

4. [KDE](https://lmfit.github.io/lmfit-py/model.html)
    1. Convolve a Gaussian kernel (bandwidth set with FD method by default) with each datum for each reference population (`R_C` and `R_N`) to produce two distribution templates: `kde_R_C` and `kde_R_N`.
    2. Create a new model which is the weighted sum of these two distributions (initialise to equal amplitudes: `amp_R_C = amp_R_N = 1`): `y = amp_R_C * kde_R_C + amp_R_N * kde_R_N`.
    3. Convolve the same kernel with the unknown mixture data to form `kde_M`.
    4. Fit the combined model to the smoothed mixture allowing the amplitude of each reference distribution to vary. Optimise with the least squares algorithm.
    5. `p_hat_C = amp_R_C / (amp_R_C + amp_R_N)`.

    ```python
    def fit_KDE_model(Mix, bins, model, params_mix, kernel, method='leastsq'):
        x = bins["centers"]
        bw = bins['width']
        kde_mix = KernelDensity(kernel=kernel, bandwidth=bw).fit(Mix[:, np.newaxis])
        res_mix = model.fit(np.exp(kde_mix.score_samples(x[:, np.newaxis])),
                            x=x, params=params_mix, method=method)
        amp_R_C = res_mix.params['amp_1'].value
        amp_R_N = res_mix.params['amp_2'].value
        return amp_R_C / (amp_R_C + amp_R_N)
    ```

Note: The population of cases is sometimes referred to as Reference 1 and the population of non-cases referred to as Reference 2. Accordingly these notations:
```
R_C == Ref1, R_N == Ref2
p_C == p1, p_N == p2
```
may be used interchangeably. 

Methods Summary
---------------

|               | Means                 | Excess                | EMD                   | KDE                   |
| ------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Advantages    | Trivial to calculate. | Trivial to calculate. | Computationally cheap. | Accurate |
|               | Relies only on summary statistics. | Relies only on summary statistics. | ... | ... |
|               | Works for even highly overlapping distributions. |    |   |
| ------------- | --------------------- | --------------------- | ----------------- | ----------------- |
| Disadvantages | Inaccurate when data is not normally distributed. | Inaccurate when the distributions significantly overlap | Requires a choice of bin size. | Requires a choice of bandwidth and kernel. |
|               |                       |                       | Computationally expensive. |

Citation
--------

If you use this software, please cite our publication:

```bibtex
@article {Evans2020.02.20.20025528,
	author = {Evans, Benjamin D. and S{\l}owi{\'n}ski, Piotr and Hattersley, Andrew T. and Jones, Samuel E. and Sharp, Seth and Kimmitt, Robert A. and Weedon, Michael N. and Oram, Richard A. and Tsaneva-Atanasova, Krasimira and Thomas, Nicholas J.},
	title = {Estimating population level disease prevalence using genetic risk scores},
	elocation-id = {2020.02.20.20025528},
	year = {2021},
	doi = {10.1101/2020.02.20.20025528},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2021/02/27/2020.02.20.20025528},
	eprint = {https://www.medrxiv.org/content/early/2021/02/27/2020.02.20.20025528.full.pdf},
	journal = {medRxiv}
}
```
