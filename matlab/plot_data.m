ref_C=scores(scores(:,1)==1,2); %assign scores to cohorts
ref_N=scores(scores(:,1)==2,2); %assign scores to cohorts
mix=scores(scores(:,1)==3,2); %assign scores to cohorts

all_socres=[ref_C; ref_N; mix];
all_ref=[ref_C; ref_N];

bin_min=min(all_socres);
bin_max=max(all_socres);

[~,edges] = histcounts(all_ref,'BinMethod','fd','BinLimits',[bin_min,bin_max]);
bw=edges(2)-edges(1);
bc=edges(1:end-1)+bw/2;
bc=[bin_min, bc, bin_max];

kde_dist_C=ksdensity(ref_C,bc,'Function','pdf','Bandwidth',bw);
kde_dist_N=ksdensity(ref_N,bc,'Function','pdf','Bandwidth',bw);
kde_mix=ksdensity(mix,bc,'Function','pdf','Bandwidth',bw);

figure(1)
clf
histogram(ref_C,edges,'Normalization','PDF','FaceColor','#e7745b','FaceAlpha',0.5,'EdgeColor','w','LineWidth',1)
hold on
histogram(ref_N,edges,'Normalization','PDF','FaceColor','#6f92f3','FaceAlpha',0.5,'EdgeColor','w','LineWidth',1)
histogram(mix,edges,'Normalization','PDF','FaceColor','#bbbbbb','FaceAlpha',0.75,'EdgeColor','w','LineWidth',1)
plot(bc,kde_dist_C,'LineWidth',2,'Color','#e7745b','handlevisibility','off')
plot(bc,kde_dist_N,'LineWidth',2,'Color','#6f92f3','handlevisibility','off')
plot(bc,kde_mix,'LineWidth',2,'Color','#bbbbbb','handlevisibility','off')

xlabel('GRS')
box off
set(gca,'TickDir','out');
legend({'R_C','R_N','M'},'Location','best','box','off')