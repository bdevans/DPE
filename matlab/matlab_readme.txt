Developed and tested on Matlab 2020b. Code requires 'Statistics and Machine Learning Toolbox'. 
With questions and comments regarding the matlab code, please contact: p.m.slowinski@exeter.ac.uk

To run the  example:
1.) change the Matlab working directory to the directory containing this readme file
2.) open and run example_script.m
 - to see performance for other prevalence values change true_pC parameter on line 6 of example_script.m
 - to generate data with different distributions, open and modify generate_example_data.m
3.) to plot the scores run plot_data.m

Please note that small differences between Matlab and Python example outputs are due to different 
pseudorandom number generator (PRNG) implementations. The example_script.m will run for around 400 sec. 
(timed on macOS 10.14.6 with Intel Core i7, 2.2 GHz processor - single core performance) and will 
produce the following output in Matlab's Command Window (Elapsed time values will change depending
on the system):

running
Elapsed time is 11.738661 seconds.
Elapsed time is 6.805715 seconds.
Elapsed time is 43.023635 seconds.
Elapsed time is 317.615285 seconds.
 
True pC := 0.25
===================================================================
EXCESS: pC (95% C):= 0.182 (0.1904, 0.2116)
-------------------------------------------------------------------
MEANS: pC (95% C):= 0.24279 (0.20468, 0.2829)
-------------------------------------------------------------------
EMD: pC (95% C):= 0.24313 (0.20581, 0.28332)
-------------------------------------------------------------------
KDE: pC (95% C):= 0.24843 (0.2047, 0.29339)
