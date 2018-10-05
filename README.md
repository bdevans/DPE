Distribution Proportion Estimation
==================================

Evaluation of algorithms for estimating the proportions of Type I (and Type II) diabetics in a mixture population based on genetic risk scores.

Execution
---------

1. Run `bootstrap.py` on a multicore server. This is best done in a Docker container (using `screen` to avoid disconnection problems) as follows:
    1. `docker build -t dpe https://git.exeter.ac.uk/bdevans/DPE.git`
    2. `screen`
    3. `docker run -it -v dpe:/usr/dpe/results --name dpe dpe`
    4. To detach, press `Ctrl+a` followed by `d`
2. Collect results and run `plots.py`.
    0. Use `screen -r` to list the detached sessions
    1. `screen -r 10654.pts-8.thuemorse`
    2. `docker cp dpe:/usr/dpe/results .`
    3. `scp -r results ben@144.173.111.1:/Users/ben/EXE/repos/DPE/results`

Methods
-------

We bootstrap (i.e. sample with replacement) from the available data, systematically varying sample size and mixture proportion. We then apply the following methods to yield estimates of the true proportion and from those, calculate the errors throughout the bootstrap parameter space.

1. Means

    ```python
    proportion_of_T1 = abs((RM.mean()-T2_mean)/(T1_mean-T2_mean))
    ```

2. [Excess](https://www.thelancet.com/journals/landia/article/PIIS2213-8587(17)30362-5/fulltext)

    ```python
    medians = {'T1GRS': 0.23137931525707245, 'T2GRS': 6.78826}

    number_low = len(RM[RM <= population_median])
    number_high = len(RM[RM > population_median])
    proportion_T1 = (number_high - number_low)/len(RM)
    ```
    Should this be abs? With the assumptions from the Lancet it is not necessary to use abs... but I guess the definition would always be case specific (it'd depend on the relation between the two reference distributions and general population)...

    a. The excess/subtraction method counts points above and below threshold `population_median`

    b. The threshold is based on statistical properties of reference GRS (Wellcome Trust Case Control Consortium cohort) and statistical properties of the general population (distribution of GRS in the general population);

    c. The Lancet analysis was possible because: "The type 1 genetic risk score in the general population has the same distribution [statistically ?] and median as [the type 1 genetic risk score in] the type 2 diabetes population." and "Almost all (96%) individuals with type 1 diabetes in the Wellcome Trust Case Control Consortium cohort have [the type 1] genetic risk score above the 50th centile of [the type 1 genetic risk score in] the type 2 diabetes cohort";

    d. ...


3. [EMD](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)
    ```math
    EMD(P,Q)={\frac {\sum _{i=1}^{m}\sum _{j=1}^{n}f_{i,j}d_{i,j}}{\sum _{i=1}^{m}\sum _{j=1}^{n}f_{i,j}}}
    ```

    ```python
    max_emd = bin_edges[-1] - bin_edges[0]

    # Interpolate the cdfs at the same points for comparison
    x_T1 = [bins[tag]['min'], *sorted(T1), bins[tag]['max']]
    y_T1 = np.linspace(0, 1, len(x_T1))
    (iv, ii) = np.unique(x_T1, return_index=True)
    i_CDF_1 = np.interp(bin_centers, iv, y_T1[ii])

    x_T2 = [bins[tag]['min'], *sorted(T2), bins[tag]['max']]
    y_T2 = np.linspace(0, 1, len(x_T2))
    (iv, ii) = np.unique(x_T2, return_index=True)
    i_CDF_2 = np.interp(bin_centers, iv, y_T2[ii])

    # EMDs computed with interpolated CDFs
    i_EMD_21 = sum(abs(i_CDF_2-i_CDF_1)) * bins[tag]['width'] / max_emd

    # For each bootstrap mixture
        # Interpolated cdf (to compute EMD)
        x = [bins[tag]['min'], *np.sort(RM), bins[tag]['max']]
        y = np.linspace(0, 1, num=len(x), endpoint=True)
        (iv, ii) = np.unique(x, return_index=True)
        si_CDF_3 = np.interp(bin_centers, iv, y[ii])

        # Compute EMDs
        i_EMD_31 = sum(abs(si_CDF_3-i_CDF_1)) * bin_width / max_emd
        i_EMD_32 = sum(abs(si_CDF_3-i_CDF_2)) * bin_width / max_emd

    # Normalise
    norm_mat_EMD_31 = mat_EMD_31 / i_EMD_21
    norm_mat_EMD_32 = mat_EMD_32 / i_EMD_21

    error_EMD_T1 = average(1-EMD_31, axis=2) - PROPORTIONS_T1D
    error_EMD_T2 = average(1-EMD_32, axis=2) - PROPORTIONS_T2D

    relative_error_EMD_T1 = 100*error_EMD_T1/PROPORTIONS_T1D
    relative_error_EMD_T2 = 100*error_EMD_T2/PROPORTIONS_T2D
    max_relative_error_EMD = np.maximum(np.abs(relative_error_EMD_T1),
                                        np.abs(relative_error_EMD_T2))

    max_abs_error_EMD = np.maximum(np.abs(error_EMD_T1), np.abs(error_EMD_T2))
    ```

4. [KDE](https://lmfit.github.io/lmfit-py/model.html)
    1. Convolve a Gaussian kernel (T1GRS bandwidth=0.005) with each datum for each reference population (T1 diabetics and T2 diabetics) to produce two distribution templates: `kde_T1` and `kde_T2`.
    2. Create a new model which is the weighted sum of these two distributions (initialise to equal amplitudes: `amp_T1 = amp_T2 = 1`).
    3. Convolve the same kernel with the unknown mixture data.
    4. Fit the combined model to the smoothed mixture allowing the amplitude of each reference distribution to vary. Optimise with the least squares algorithm.
    5. `proportion_T1 = amp_T1/(amp_T1+amp_T2)`.

    ```python
    # Fit reference populations
    for data, label in zip([T1, T2], labels):
        kdes[label] = {}
        X = data[:, np.newaxis]
        kde = KernelDensity(kernel=kernel, bandwidth=bin_width).fit(X)
        kdes[label] = kde

    def fit_KDE(RM, model, params_mix, kernel, bins):
        x_KDE = np.linspace(bins[tag]['min'], bins[tag]['max'], len(RM)+2)
        mix_kde = KernelDensity(kernel=kernel, bandwidth=bins[tag]['width']).fit(RM[:, np.newaxis])
        res_mix = model.fit(np.exp(mix_kde.score_samples(x_KDE[:, np.newaxis])), x=x_KDE, params=params_mix)
        amp_T1 = res_mix.params['amp_T1'].value
        amp_T2 = res_mix.params['amp_T2'].value
        return amp_T1/(amp_T1+amp_T2)
    ```

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


Discussion points to be agreed
------------------------------

- [x] Use adjusted excess method i.e. pr_Excess /= 0.92
- [ ] Should the Excess take as Ref2, the closest to the mixture?
- [ ] Should the Excess flip the references for the heatmap?
- [ ] How to calculate "pseudo ground truth" for Diabetes data
- [ ] How to generate a single proportion from EMD method
- [ ] Supplementary figure showing the Excess breakdown for different degrees of overlap?
- [ ] Should the grid search plots colour +/- error?


TODO
----

- [ ] [Remove sensitive data](https://help.github.com/articles/removing-sensitive-data-from-a-repository/) from the repository before publishing e.g. `biobank_mix_WTCC_ref.csv`.
- [ ] Why is the population_median close to but not equal to the Ref1_median?
- [x] Summary table of pros and cons for each method
- [ ] Apply excess and kde to mixture plots for mean and CI
- [x] Include equation for clinical characteristics from Thomas et al. Lancet Diabetes and Endocrinology
- [ ] Plot residuals for KDE to help identify "third" groups
- [ ] Ensure all relevant data access forms are signed
- [ ] Adapt EMD to be non-parametric
- [ ] Adapt KDE to be non-parametric
- [ ] Reference for the 8% Excess underestimate?
- [ ] The Wellcome Trust requires that the research is made open access: https://www.exeter.ac.uk/research/openresearch/payingopenaccess/

Continuous characteristics (eg, BMI) were derived by use of the mean value of the low susceptibility group (type 2 diabetes) and the mean of the high susceptibility group (combined type 1 and type 2 diabetes) to calculate a mean for the group with type 1 diabetes. For example, for BMI:

```math
\bar{x}_{T1D}^{BMI} = \frac{n_H \bar{x}_H^{BMI} - n_L \bar{x}_L^{BMI}}{n_{T1D}}
```

where $`n_H`$, $`n_L`$, and $`n_{T1D}`$ represent the number of individuals and $`\bar{x}_H^{BMI}`$, $`\bar{x}_L^{BMI}`$, and $`\bar{x}_{T1D}^{BMI}`$ represent the mean BMI of the high, low, and excess groups, respectively.

Display Items
-------------

1. Bootstrap figure
2. Methods illustration (including excess) consistent layout with results
3. Type I Diabetes: Results (error & s.d. for each method) [2x2]
4. Violin plot or swarm in box plot (estimate with confidence intervals)
    - Compute estimates for each method for each real data set [3x3] then fix each estimate to bootstrap around (drawing from reference populations) in order to compute CIs for each method on each data set.
5. Table of pros and cons

### Supplementary Figures

1. Bootstraps (as for Fig. 3) for e.g. Depression, Menieres, Renal.
2. Mixture histograms with estimates and confidence intervals for each method
3. Distributions of Ref1, Ref2 and Mix for each dataset
