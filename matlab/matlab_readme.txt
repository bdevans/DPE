Developed and tested on Matlab 2020b. Code requires 'Statistics and Machine Learning Toolbox'. 

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
Elapsed time is 11.340442 seconds.
Elapsed time is 6.623568 seconds.
Elapsed time is 42.111663 seconds.
Elapsed time is 309.899719 seconds.
 
True pC := 0.25
===================================================================
EXCESS:	pC (95% C):= 0.18 (0.187, 0.22)	Bias:= -0.0484
-------------------------------------------------------------------
MEANS:	pC (95% C):= 0.246 (0.209, 0.285)	Bias:= -0.000927
-------------------------------------------------------------------
EMD:	pC (95% C):= 0.245 (0.209, 0.286)	Bias:= -0.00204
-------------------------------------------------------------------
KDE:	pC (95% C):= 0.244 (0.205, 0.294)	Bias:= -0.00277
