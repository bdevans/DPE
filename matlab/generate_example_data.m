function scores=generate_example_data(pC,sample_size,seed)
% INPUT:
% pC - proportion of non-cases ref1
% sample_size - common size of each sample
% seed - seed of the pseudorandom number generator, set for reproducibility
% OUTPUT:
% scores - matrix with artificial scores. 
%          1st column is a label 1 - ref_C, 2 - ref_N, 3 - mixture 
%          2nd column has random numbers (artificial scores)

rng(seed);
n_ref_C=sample_size; % size of cases sample
n_ref_N=sample_size; % size of non-cases sample
n_mix=sample_size; % total size of mixture sample

%generate ref1
ref1=1+0.2*randn(n_ref_C,1); %normal distribution with mean=1 and std=0.2

%generate ref2
bumpN=ceil(0.1*n_ref_N); %number of samples used to generate side bump in ref2
ref2=[(1.2+0.2*randn(n_ref_N-bumpN,1)); (1.6+0.05*randn(bumpN,1))]; 
% mix of two normal distributions:
% with mean=1.2 and std=0.2 and with mean=1.6 and std=0.05

% generate mixture
n_mix_C=ceil(pC*n_mix); %number of samples from cases (ref1)
n_mix_N=n_mix-n_mix_C; %number of samples from non-cases (ref2)
mix=[ref1(randi(n_ref_C,n_mix_C,1)); ref2(randi(n_ref_N,n_mix_N,1))];
%mixture is sampled from ref1 and ref2

%saving
scores(1:n_ref_C,1)=ones(1,n_ref_C);
scores((n_ref_C+1):(n_ref_C+n_ref_N),1)=2*ones(1,n_ref_N);
scores((n_ref_C+n_ref_N+1):(n_ref_C+n_ref_N+n_mix),1)=3*ones(1,n_mix);
scores(1:n_ref_C,2)=ref1;
scores((n_ref_C+1):(n_ref_C+n_ref_N),2)=ref2;
scores((n_ref_C+n_ref_N+1):(n_ref_C+n_ref_N+n_mix),2)=mix;