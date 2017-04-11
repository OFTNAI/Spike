cd('/Users/dev/Documents/Spike_aki/Spike2_trans/output/5.1a--FFLATFB_EP20')

maxPair = 500000;

IRs0_u = load('SingleCellInfo_l0Untrained_max.csv');
IRs1_u = load('SingleCellInfo_l1Untrained_max.csv');
IRs2_u = load('SingleCellInfo_l2Untrained_max.csv');
IRs3_u = load('SingleCellInfo_l3Untrained_max.csv');
IRs0_t = load('SingleCellInfo_l0Trained_max.csv');
IRs1_t = load('SingleCellInfo_l1Trained_max.csv');
IRs2_t = load('SingleCellInfo_l2Trained_max.csv');
IRs3_t = load('SingleCellInfo_l3Trained_max.csv');


IPs0_u = load('SingleCellInfo_SpikePairs_l0Untrained_max.csv');
IPs0_u = [IPs0_u;zeros(maxPair-length(IPs0_u),1)];
IPs1_u = load('SingleCellInfo_SpikePairs_l1Untrained_max.csv');
IPs1_u = [IPs1_u;zeros(maxPair-length(IPs1_u),1)];
IPs2_u = load('SingleCellInfo_SpikePairs_l2Untrained_max.csv');
IPs2_u = [IPs2_u;zeros(maxPair-length(IPs2_u),1)];
IPs3_u = load('SingleCellInfo_SpikePairs_l3Untrained_max.csv');
IPs3_u = [IPs3_u;zeros(maxPair-length(IPs3_u),1)];

IPs0_t = load('SingleCellInfo_SpikePairs_l0Trained_max.csv');
IPs0_t = [IPs0_t;zeros(maxPair-length(IPs0_t),1)];
IPs1_t = load('SingleCellInfo_SpikePairs_l1Trained_max.csv');
IPs1_t = [IPs1_t;zeros(maxPair-length(IPs1_t),1)];
IPs2_t = load('SingleCellInfo_SpikePairs_l2Trained_max.csv');
IPs2_t = [IPs2_t;zeros(maxPair-length(IPs2_t),1)];
IPs3_t = load('SingleCellInfo_SpikePairs_l3Trained_max.csv');
IPs3_t = [IPs3_t;zeros(maxPair-length(IPs3_t),1)];


cd('/Users/dev/Documents/Spike_aki/Spike2_trans/MatLabPG/');

hFig = figure(1);
set(hFig, 'Position', [50 50 1000 400])
subplot(2,4,1);
plot(IRs0_u,'k:','LineWIdth',2);
hold on
plot(IRs0_t,'k','LineWIdth',2);
xlim([1,1000])
ylim([-0.1,1.1])
ylabel('Information [bit]');
xlabel('Cell Rank');
title('1st Layer');

subplot(2,4,2);
plot(IRs1_u,'k:','LineWIdth',2);
hold on
plot(IRs1_t,'k','LineWIdth',2);
xlim([1,1000])
ylim([-0.1,1.1])
ylabel('Information [bit]');
xlabel('Cell Rank');
title('2nd Layer');

subplot(2,4,3);
plot(IRs2_u,'k:','LineWIdth',2);
hold on
plot(IRs2_t,'k','LineWIdth',2);
xlim([1,1000])
ylim([-0.1,1.1])
ylabel('Information [bit]');
xlabel('Cell Rank');
title('3rd Layer');

subplot(2,4,4);
plot(IRs3_u,'k:','LineWIdth',2);
hold on
plot(IRs3_t,'k','LineWIdth',2);
xlim([1,1000])
ylim([-0.1,1.1])
ylabel('Information [bit]');
xlabel('Cell Rank');
title('4th Layer');

%hFig = breakxaxis([200 600]);







subplot(2,4,5);
plot(IPs0_u,'k:','LineWIdth',2);
hold on
plot(IPs0_t,'k','LineWIdth',2);
ylim([-0.1,1.1])
xlim([1,97000])
hFig = breakxaxis([10000 97000-8000]);
ylabel('Information [bit]');
xlabel('Spike-Pair Rank');


subplot(2,4,6);
plot(IPs1_u,'k:','LineWIdth',2);
hold on
plot(IPs1_t,'k','LineWIdth',2);
ylim([-0.1,1.1])
xlim([1,320000])
hFig = breakxaxis([10000 320000-8000]);
ylabel('Information [bit]');
xlabel('Spike-Pair Rank');

subplot(2,4,7);
plot(IPs2_u,'k:','LineWIdth',2);
hold on
plot(IPs2_t,'k','LineWIdth',2);
ylim([-0.1,1.1])
xlim([1,480000])
hFig = breakxaxis([10000 472000]);
ylabel('Information [bit]');
xlabel('Spike-Pair Rank');

subplot(2,4,8);
plot(IPs3_u,'k:','LineWIdth',2);
hold on
plot(IPs3_t,'k','LineWIdth',2);
ylim([-0.1,1.1]);
xlim([1,470000])
hFig = breakxaxis([10000 462000]);
ylabel('Information [bit]');
xlabel('Spike-Pair Rank');
