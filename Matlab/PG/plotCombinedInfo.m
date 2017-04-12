% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.1--FF_stdp_0.005')
% IRs1_u=load('SingleCellInfo_l3Untrained_max.csv');
% IRs1_t=load('SingleCellInfo_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.2--FF_FB_stdp_0.005')
% IRs2_u=load('SingleCellInfo_l3Untrained_max.csv');
% IRs2_t=load('SingleCellInfo_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.3--FF_LAT_stdp_0.005')
% IRs3_u=load('SingleCellInfo_l3Untrained_max.csv');
% IRs3_t=load('SingleCellInfo_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.4--FF_FB_LAT_stdp_0.005')
% IRs4_u=load('SingleCellInfo_l3Untrained_max.csv');
% IRs4_t=load('SingleCellInfo_l3Trained_max.csv');
figure
subplot(1,2,1);
plot(IRs1_t,'k:','LineWidth',2);
hold on
plot(IRs1_u,':','Color', [0.5 0.5 0.5]);

plot(IRs2_t,'k.-','LineWidth',2);
plot(IRs2_u,'.-','Color', [0.5 0.5 0.5]);

plot(IRs3_t,'k--','LineWidth',2);
plot(IRs3_u,'--','Color', [0.5 0.5 0.5]);

plot(IRs4_t,'k-','LineWidth',2);
plot(IRs4_u,'-','Color', [0.5 0.5 0.5]);



xlim([1 500])
ylim([-log2(3)*0.05 log2(3)*1.05]);
title('Single Cell Information');
ylabel('Information [bit]');
xlabel('Cell Rank');


subplot(1,2,2);

% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.1--FF_stdp_0.005')
% IRp1_u=load('SingleCellInfo_SpikePairs_l3Untrained_max.csv');
% IRp1_t=load('SingleCellInfo_SpikePairs_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.2--FF_FB_stdp_0.005')
% IRp2_u=load('SingleCellInfo_SpikePairs_l3Untrained_max.csv');
% IRp2_t=load('SingleCellInfo_SpikePairs_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.3--FF_LAT_stdp_0.005')
% IRp3_u=load('SingleCellInfo_SpikePairs_l3Untrained_max.csv');
% IRp3_t=load('SingleCellInfo_SpikePairs_l3Trained_max.csv');
% cd('/Volumes/Transcend/Spike_aki/Spike/output/1.4--FF_FB_LAT_stdp_0.005')
% IRp4_u=load('SingleCellInfo_SpikePairs_l3Untrained_max.csv');
% IRp4_t=load('SingleCellInfo_SpikePairs_l3Trained_max.csv');

plot(IRp1_t,'k:','LineWidth',2);
hold on
plot(IRp1_u,':','Color', [0.5 0.5 0.5]);

plot(IRp2_t,'k.-','LineWidth',2);
plot(IRp2_u,'.-','Color', [0.5 0.5 0.5]);

plot(IRp3_t,'k--','LineWidth',2);
plot(IRp3_u,'--','Color', [0.5 0.5 0.5]);

plot(IRp4_t,'k-','LineWidth',2);
plot(IRp4_u,'-','Color', [0.5 0.5 0.5]);



xlim([1 5000])
ylim([-log2(3)*0.05 log2(3)*1.05]);
title('Single Spike-Pair Information');
ylabel('Information [bit]');
xlabel('Cell Rank');