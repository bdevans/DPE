function [p_I,CI_BCa,B,prv_bts,prv_jckknf]=est_prp_kde(ref1,ref2,mix,CI_centile)
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

all_socres=[ref1; ref2; mix];
all_ref=[ref1; ref2];

bin_min=min(all_socres);
bin_max=max(all_socres);

[~,edges] = histcounts(all_ref,'BinMethod','fd','BinLimits',[bin_min,bin_max]);
bw=edges(2)-edges(1);
bc=edges(1:end-1)+bw/2;

kde_dist1=ksdensity(ref1,bc,'Function','pdf','Bandwidth',bw);
kde_dist2=ksdensity(ref2,bc,'Function','pdf','Bandwidth',bw);
kde_mix=ksdensity(mix,bc,'Function','pdf','Bandwidth',bw);

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Display','off');

fun=@(pp,r)pp(1)*r(:,1)+pp(2)*r(:,2);
prv = lsqcurvefit(fun,[1 1],[kde_dist1' kde_dist2'],kde_mix',[],[],options);
p_I=prv/sum(prv);

if p_I(1)<0
    p_I(1)=0;
    p_I(2)=1;
elseif p_I(1)>1
    p_I(1)=1;
    p_I(2)=0;
end

N_ref1=numel(ref1);
N_ref2=numel(ref2);
N_mix=numel(mix);


% jackknife estimates of pre to later estimate Empirical influence function
prv_jckknf=NaN(1,N_mix);

for i_jckknf=1:N_mix
    jckk_mix=mix;
    jckk_mix(i_jckknf)=[];
    kde_mix=ksdensity(jckk_mix,bc,'Function','pdf','Bandwidth',bw);
    prv = lsqcurvefit(fun,[1 1],[kde_dist1' kde_dist2'],kde_mix',[],[],options);
    p_I_jckknf=prv/sum(prv);
    
    if  p_I_jckknf(1)<0
        p_I_jckknf(1)=0;
    elseif p_I_jckknf(1)>1
        p_I_jckknf(1)=1;
    end
    
    prv_jckknf(i_jckknf)=p_I_jckknf(1);
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
        
        kde_mix=ksdensity(mixture_rs,bc,'Function','pdf','Bandwidth',bw);
        prv = lsqcurvefit(fun,[1 1],[kde_dist1' kde_dist2'],kde_mix',[],[],options);
        prv=prv/sum(prv);
        
        if  prv(1)<0
            prv(1)=0;
        elseif prv(1)>1
            prv(1)=1;
        end
        
        prv_bts(i_mix,i_bts)=prv(1);
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
