figure
subplot(1,2,1);
plot(IRs4_u,'-','Color',[0.5,0.5,0.5],'LineWidth',2)
hold on
plot(IRs1_t,'k:','LineWidth',2)
plot(IRs2_t,'k-.','LineWidth',2)
plot(IRs3_t,'k--','LineWidth',2)
plot(IRs4_t,'k-','LineWidth',2)
xlim([1 300])
ylim([-log2(3)*0.05 log2(3)*1.05])
title('Single Cell Information');
ylabel('Information [bit]')
xlabel('Cell Rank')


subplot(1,2,2);
hold off
plot(IRp4_u,'-','Color',[0.5,0.5,0.5],'LineWidth',2)
hold on
plot(IRp1_t,'k:','LineWidth',2)
plot(IRp2_t,'k-.','LineWidth',2)
plot(IRp3_t,'k--','LineWidth',2)
plot(IRp4_t,'k-','LineWidth',2)
xlim([1 55250])
ylim([-log2(3)*0.05 log2(3)*1.05])
title('Single Spike-Pair Information');
ylabel('Information [bit]')
xlabel('Spike-Pair Rank')
h = breakxaxis([1400 53500]);





subplot(1,2,1);
% plot(IRs4_u,'-','Color',[0.5,0.5,0.5],'LineWidth',2)
% hold on
hold off
plot(IRp4_125_t,'k:','LineWidth',2)
hold on
plot(IRp4_025_t,'k--','LineWidth',2)
plot(IRp4_t,'k-','LineWidth',2)
xlim([1 59250])
ylim([-log2(3)*0.05 log2(3)*1.05])
title('Single Spike-Pair Information');
ylabel('Information [bit]')
xlabel('Spike-Pair Rank')
h = breakxaxis([1400 58700]);

subplot(1,2,2);
% plot(IRs4_u,'-','Color',[0.5,0.5,0.5],'LineWidth',2)
% hold on
hold off
plot(IRp4_2Con_t,'k--','LineWidth',2)
hold on
plot(IRp4_t,'k-','LineWidth',2)
xlim([1 227500])
ylim([-log2(3)*0.05 log2(3)*1.05])
title('Single Spike-Pair Information');
ylabel('Information [bit]')
xlabel('Spike-Pair Rank')
h = breakxaxis([5000 224000]);


