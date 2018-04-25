Diabetes Proportion Estimation
==============================

Evaluation of algorithms for estimating the proportions of Type I (and Type II) diabetics in a mixture population based on genetic risk scores.

Methods
-------

1. Means

    ```python
    proportion_of_T1 = abs(RM.mean()-T2_mean)/(T1_mean-T2_mean)
    ```

2. [Excess](https://www.thelancet.com/journals/landia/article/PIIS2213-8587(17)30362-5/fulltext)

    ```python
    number_low = len(RM[RM <= population_median])
    number_high = len(RM[RM > population_median])
    difference = number_high - number_low
    proportion_T1 = difference/(2*number_low+difference)
    ```
Should this be abs?

3. [EMD](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)

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
    1. Convolve a Gaussian kernel (T1GRS bandwidth=0.005) with each datum for each reference population (T1 diabetics and T2 diabetics) to produce two distribution templates.
    2. Create a new model which is the weighted sum of these two distributions.
    3. Convolve the same kernel with the unknown mixture data.
    4. Fit the combined model to the smoothed mixture allowing the amplitude of each reference distribution to vary. Optimise with the least squares algorithm.
    5. Return amp_T1/(amp_T1+amp_T2).


TODO
----

* [Remove sensitive data](https://help.github.com/articles/removing-sensitive-data-from-a-repository/) from the repository before publishing e.g. `biobank_mix_WTCC_ref.csv`.
