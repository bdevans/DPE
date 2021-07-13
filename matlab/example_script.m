clear %clear all variables
rng(42) %seed set for reproducibility not necessery for real applications

%load sample
%all samples are generated using the generate_example_data.m function
true_pC=0.25; sample_size=5000; seed=42;
scores=generate_example_data(true_pC,sample_size,seed);
%1st column is a label. 1 - ref_C, 2 - ref_N, 3 - mixture
%2nd column has random numbers (artificial scores)

ref_C=scores(scores(:,1)==1,2); %assign scores to cohorts
ref_N=scores(scores(:,1)==2,2); %assign scores to cohorts
mix=scores(scores(:,1)==3,2); %assign scores to cohorts
CI_centile=0.05; %centile for calculating the confidence intervals from bootstrapped estimates

%each function is set in exactly the same way as in the manuscript
%100 mixture
%1000 bootstraps
%Freedman-Diaconis bins
%95% confidence intervals
%each function has the same output
%p_I - vector of point estimates p_I(1) is pC, p_I(2) is pN
%CI - vector of 95% confidence intervals CI(1,2) for p_I(1), CI(3,4) for p_I(2)
%B - median bias of the method estimated from the bootstrap, median(bootstraps)-p_I(1)
%
%Please note that small differences between Matlab and Python example outputs 
%are due to different pseudorandom number generator (PRNG) implementations.

clc
disp('running')
tic
[p_I_xcs,CI_xcs,B_xcs]=est_prp_excess(ref_C,ref_N,mix,CI_centile);
toc
tic
[p_I_mn,CI_mn,B_mn]=est_prp_mean(ref_C,ref_N,mix,CI_centile);
toc
tic
[p_I_emd,CI_emd,B_emd]=est_prp_emd(ref_C,ref_N,mix,CI_centile);
toc
tic
[p_I_kde,CI_kde,B_kde]=est_prp_kde(ref_C,ref_N,mix,CI_centile);
toc
%%
disp(' ')
disp(['True pC := ' num2str(true_pC,3)])
disp('===================================================================')
disp(['EXCESS:' 9 'pC (95% CI):= ' num2str(p_I_xcs(1),3) ' (' num2str(CI_xcs(1),3) ', ' num2str(CI_xcs(2),3) ')' 9 'Bias:= '  num2str(B_xcs,3)])
disp('-------------------------------------------------------------------')
disp(['MEANS:' 9 'pC (95% CI):= ' num2str(p_I_mn(1),3) ' (' num2str(CI_mn(1),3) ', ' num2str(CI_mn(2),3) ')' 9 'Bias:= '  num2str(B_mn,3)])
disp('-------------------------------------------------------------------')
disp(['EMD:' 9 'pC (95% CI):= ' num2str(p_I_emd(1),3) ' (' num2str(CI_emd(1),3) ', ' num2str(CI_emd(2),3) ')' 9 'Bias:= '  num2str(B_emd,3)])
disp('-------------------------------------------------------------------')
disp(['KDE:' 9 'pC (95% CI):= ' num2str(p_I_kde(1),3) ' (' num2str(CI_kde(1),3) ', ' num2str(CI_kde(2),3) ')' 9 'Bias:= '  num2str(B_kde,3)])
