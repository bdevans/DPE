%% bootstrap
data=importdata('data_for_piotr.csv');
for_analysis=data.data(:,2:3);

k=0.005;
bin_edges=0.095:k:0.35;
bin_centers=(0.095:k:(0.35-k))+k/2;
bin_width=bin_edges(2)-bin_edges(1);
max_emd=bin_edges(end)-bin_edges(1);

ht1=for_analysis(:,1);
ht1=ht1(for_analysis(:,2)==1);
ht2=for_analysis(:,1);
ht2=ht2(for_analysis(:,2)==2);
ht3=for_analysis(:,1);
ht3=ht3(for_analysis(:,2)==3);

DH1=linspace(0,1,numel(ht1));
sDH1=sort(ht1);
x=[0.095 sDH1' 0.35];
y=linspace(0,1,numel(DH1)+2);
[iv,ii,~] = unique(x);
i_cdf1=interp1(iv,y(ii),bin_centers);

DH2=linspace(0,1,numel(ht2));
sDH2=sort(ht2);
x=[0.095 sDH2' 0.35];
y=linspace(0,1,numel(DH2)+2);
[iv,ii,~] = unique(x);
i_cdf2=interp1(iv,y(ii),bin_centers);

DH3=linspace(0,1,numel(ht3));
sDH3=sort(ht3);
x=[0.095 sDH3' 0.35];
y=linspace(0,1,numel(DH3)+2);
[iv,ii,~] = unique(x);
i_cdf3=interp1(iv,y(ii),bin_centers);

hc1=histc(ht1,bin_edges);
hc2=histc(ht2,bin_edges);
hc3=histc(ht3,bin_edges);

% emds computed with histograms
emd_21=sum(abs(cumsum(hc2/sum(hc2))-cumsum(hc1/sum(hc1))))*bin_width*max_emd
emd_31=sum(abs(cumsum(hc3/sum(hc3))-cumsum(hc1/sum(hc1))))*bin_width*max_emd
emd_32=sum(abs(cumsum(hc2/sum(hc2))-cumsum(hc3/sum(hc3))))*bin_width*max_emd

% emds computed with interpolated cdfs
iemd_21=sum(abs(i_cdf2-i_cdf1))*bin_width*max_emd
iemd_31=sum(abs(i_cdf3-i_cdf1))*bin_width*max_emd
iemd_32=sum(abs(i_cdf3-i_cdf2))*bin_width*max_emd
%%
i=1;
max_rn=10;
emd_dev_from_fit=zeros(30,50,max_rn);
rms_dev_from_fit=zeros(30,50,max_rn);
mat_emd_31=zeros(30,50,max_rn);
mat_emd_32=zeros(30,50,max_rn);

for rn=1:max_rn
    i=1;
    tic,
    for sample=100:100:3000
        k=1;
        for prpt=0.01:0.02:0.99
            % random sample from population 1
            R1 = rand(round(sample*prpt),1); % makes vector of random numbers from uniform01 with required number of samples for a given proportion "round(sample*prpt)"
            p1 = @(R1) find(R1<DH1,1,'first'); % function with the name p1 that finds where the random number is on the [0,1] interval.
            rR1 = arrayfun(p1,R1); % apply the function p1 to all the random numbers in R1
            st1=sDH1(rR1); % get the actual values of the T1GRS from the population (the T1GRS values are sorted because in this way values can be identified with intervals on [0,1])

            % random sample from population 2
            R2 = rand(round(sample*(1-prpt)),1);
            p2 = @(R2) find(R2<DH2,1,'first');
            rR2 = arrayfun(p2,R2);
            st2=sDH2(rR2);

            % experimental cdf
            ht3s=[st1; st2];
            DH3=linspace(0,1,numel(ht3s));
            sDH3=sort(ht3s);

            % interpolated cdf (to compute emd)
            x=[0.095 sDH3' 0.35];
            y=linspace(0,1,numel(DH3)+2);
            [iv,ii,~] = unique(x);
            si_cdf3=interp1(iv,y(ii),bin_centers);

            % compute emds
            iemd_31=sum(abs(si_cdf3-i_cdf1))*bin_width*max_emd;
            iemd_32=sum(abs(si_cdf3-i_cdf2))*bin_width*max_emd;

            emd_dev_from_fit(i,k,rn)=sum(si_cdf3-((1-iemd_32/iemd_21)*i_cdf2+(1-iemd_31/iemd_21)*i_cdf1))*bin_width*max_emd; % deviations from fit measured with emd
            rms_dev_from_fit(i,k,rn)=sqrt(sum((si_cdf3-((1-iemd_32/iemd_21)*i_cdf2+(1-iemd_31/iemd_21)*i_cdf1)).^2))/numel(si_cdf3); % deviations from fit measured with rms
            mat_emd_31(i,k,rn)=iemd_31; % emds to compute proportions
            mat_emd_32(i,k,rn)=iemd_32; % emds to compute proportions

            k=k+1;
        end
        i=i+1,
    end
    toc,
end
%% deviation from fit
surf(median(emd_dev_from_fit/0.0128,3)) %normalised with the emd distance between the two orignal distributions
hold on
contour3(median(emd_dev_from_fit/0.0128,3),[0.01 0.001]*max(emd_dev_from_fit(:)/0.0128),'r','LineWidth',3)
hold off
set(gca,'xtick',linspace(1,50,21),'xticklabel',[1 5:5:99 99])
xlabel('% proportion')
set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
ylim([1 30])
ylabel('sample size')
%% error p2
input_prop=repmat(0.01:0.02:0.99,30,1,10); %set proportion
surf(100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3))
hold on
% 5% relative error contour the other proportion
contour3(100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3),[5 5],'r','LineWidth',3)
hold off
set(gca,'xtick',linspace(1,50,21),'xticklabel',[1 5:5:99 99])
xlabel('% proportion')
set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
ylim([1 30])
ylabel('sample size')
%% error p1
input_prop=repmat(0.99:-0.02:0.01,30,1,10); %set original proportion
surf(100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3))
hold on
% 5% relative error contour the other proportion
contour3(100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3),[5 5],'r','LineWidth',3)
hold off
set(gca,'xtick',linspace(1,50,21),'xticklabel',[99 99:-5:5 1])
xlabel('% proportion')
set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
ylim([1 30])
ylabel('sample size')
%% max error
input_prop=repmat(0.99:-0.02:0.01,30,1,10); %set original proportion
ers(:,:,1)=100*median(((1-mat_emd_32/iemd_21)-input_prop)./input_prop,3);
input_prop=repmat(0.01:0.02:0.99,30,1,10); %set proportion
ers(:,:,2)=100*median(((1-mat_emd_31/iemd_21)-input_prop)./input_prop,3);
surf(max(ers,[],3))
hold on
% 5% relative error contour the max error proportion
contour3(max(ers,[],3),[5 5],'r','LineWidth',3)
hold off
set(gca,'xtick',linspace(1,50,21),'xticklabel',[1 5:5:99 99])
xlabel('% proportion')
set(gca,'ytick',[1 5:5:30],'yticklabel',[100 500:500:3000],'Ydir','reverse')
ylim([1 30])
ylabel('sample size')
