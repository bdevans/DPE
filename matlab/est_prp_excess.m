function [p_I,CI_BCa,B,prv_bts,prv_jckknf]=est_prp_excess(ref1,ref2,mix,CI_centile)
%INPUT
%ref1 - 1st reference sample (typically cases)
%ref2 - 2nd reference sample (typically non-cases)
%mix - mixture sample
%CI_centile - centile for calculating the confidence intervals from bootstrapped estimates
%
%OUTPUT
%p_I - vector of point estimates p_I(1) is pC, p_I(2) is pN
%CI - vector of 0.05 confidence intervals CI(1,2) for p_I(1), CI(3,4) for p_I(2)
%B - median bias of the method estimated from the bootstrap
%
%ADDITIONAL output
%prv_bts - bootstrap values (can be used for plotting or diagnostics)
%prv_jckknf - jackknife values (can be used for plotting or diagnostics)
%contact: p.m.slowinski@exeter.ac.uk

ref1=ref1(:);
ref2=ref2(:);
mix=mix(:);

%median of the non-cases distribution
mdrn=median(ref2);

%median of the cases distribution
mdrc=median(ref1);

if mdrn>mdrc
    N1=sum(mix(:)<=mdrn);
    N2=sum(mix(:)>mdrn);
    
    prv=NaN(1,2);
    prv(1)=abs(N2-N1)/numel(mix);
    prv(2)=1-prv(1);
    
    p_I=prv;
    
    N_ref1=numel(ref1);
    N_ref2=numel(ref2);
    N_mix=numel(mix);
    
    % jackknife estimates of pre to later estimate Empirical influence function
    prv_jckknf=NaN(1,N_mix);
    
    for i_jckknf=1:N_mix
        jckk_mix=mix;
        jckk_mix(i_jckknf)=[];
        N1=sum(jckk_mix(:)<=mdrn);
        N2=sum(jckk_mix(:)>mdrn);
        prv_jckknf(i_jckknf)=abs(N2-N1)/numel(jckk_mix);
    end
    
    N_p1=round(N_mix*p_I(1));
    N_p2=N_mix-N_p1;
    
    prv_bts=NaN(100,1000);
    
    for i_mix=1:100
        mixture_gen=[ref1(randi(N_ref1,1,N_p1)); ref2(randi(N_ref2,1,N_p2))];
        
        % bootstrap 1000 times
        for i_bts=1:1000
            mixture_rs=mixture_gen(randi(N_mix,1,N_mix));
            
            N1=sum(mixture_rs(:)<=mdrn);
            N2=sum(mixture_rs(:)>mdrn);
            
            prv_bts(i_mix,i_bts)=abs(N2-N1)/N_mix;
        end
    end
%---------------------------
elseif mdrn<=mdrc % case if median R_N is bigger than median of R_C
    
    N1=sum(mix(:)>mdrn);
    N2=sum(mix(:)<=mdrn);
    prv=NaN(1,2);
    prv(1)=abs(N2-N1)/numel(mix);
    prv(2)=1-prv(1);
    
    p_I=prv;
    
    N_ref1=numel(ref1);
    N_ref2=numel(ref2);
    N_mix=numel(mix);
    
    % jackknife estimates of pre to later estimate Empirical influence function
    prv_jckknf=NaN(1,N_mix);
    
    for i_jckknf=1:N_mix
        jckk_mix=mix;
        jckk_mix(i_jckknf)=[];
        N1=sum(jckk_mix(:)>mdrn);
        N2=sum(jckk_mix(:)<=mdrn);
        prv_jckknf(i_jckknf)=abs(N2-N1)/numel(jckk_mix);
    end
    
    N_p1=round(N_mix*p_I(1));
    N_p2=N_mix-N_p1;
    
    prv_bts=NaN(100,1000);

    % 100 model mixtures
    for i_mix=1:100
        mixture_gen=[ref1(randi(N_ref1,1,N_p1)); ref2(randi(N_ref2,1,N_p2))];
        
        % bootstrap 1000 times
        for i_bts=1:1000
            mixture_rs=mixture_gen(randi(N_mix,1,N_mix));
            
            N1=sum(mixture_rs(:)>mdrn);
            N2=sum(mixture_rs(:)<=mdrn);
            prv_bts(i_mix,i_bts)=abs(N2-N1)/N_mix;
        end
    end
end

%%% BCa
stat = p_I(1);
bstat = prv_bts(:);
jstat = prv_jckknf(:);

% CI bias correction
z_0 = norminv(sum(bstat<=stat)/1e5);

% compute acceleration term, see DiCiccio and Efron (1996)
% DiCiccio, T. J. and B. Efron. (1996). Bootstrap Confidence Intervals.
% Statistical Science. 11(3): 189–212. https://doi.org/10.1214/ss/1032280214
% Empirical/  jackknife influence function
mjstat = mean(jstat); % mean
% equation 6.7 in DiCiccio, T. J. and B. Efron. (1996)
score = mjstat-jstat; % (N-1) factor cancels out in the skew %
% equation 6.6 in DiCiccio, T. J. and B. Efron. (1996) also equation 7.3 and begining of 7.4 in
% Bradley Efron (1987) Better Bootstrap Confidence Intervals,
% Journal of the American Statistical Association, 82:397, 171-185
% https://doi.org/10.1080/01621459.1987.10478410
skew = sum(score.^3) ./ (sum(score.^2).^(3/2));%  skewness of the score function
acc = skew/6;  % acceleration ignor /sqrt(N_mix) (e.g. in the end of equation 7.4
% from Bradley Efron (1987)) because we are not using moments

% transform back with bias corrected and acceleration
% with expanision for small sample sizes
% Tim C. Hesterberg (2015) What Teachers Should Know About the Bootstrap:
% Resampling in the Undergraduate Statistics Curriculum,
% The American Statistician, 69(4): 371-386. https://doi.org/10.1080/00031305.2015.1089789
cnt_l=CI_centile/2;
cnt_h=1-cnt_l;
expanded_alpha=normcdf(tinv([cnt_l cnt_h],N_mix-1)* sqrt(N_mix / (N_mix-1)));

z_alpha1 = norminv(expanded_alpha(1));
z_alpha2 = norminv(expanded_alpha(2));

pct1 = 100*normcdf(z_0 +(z_0+z_alpha1)./(1-acc.*(z_0+z_alpha1)));
pct1(z_0==Inf) = 100;
pct1(z_0==-Inf) = 0;
pct2 = 100*normcdf(z_0 +(z_0+z_alpha2)./(1-acc.*(z_0+z_alpha2)));
pct2(z_0==Inf) = 100;
pct2(z_0==-Inf) = 0;

% inverse of ECDF
CI_BCa(1) = prctile(bstat,pct1,1);
CI_BCa(2) = prctile(bstat,pct2,1);
CI_BCa=sort(CI_BCa);
CI_BCa(3)=1-CI_BCa(2);
CI_BCa(4)=1-CI_BCa(1);

%%% median bias
B=median(bstat)-p_I(1);
end
