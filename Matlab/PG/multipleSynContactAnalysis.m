experimentName = '1.5--FF_FB_LAT_stdp_0.005_nCon2';

ExcitDim = 64;
InhibDim = 32;
nLayers = 4;
N = (ExcitDim*ExcitDim+InhibDim*InhibDim)*nLayers;% N: num neurons
timestep = 0.00002;


fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights_Initial.bin']);
weights_u_loaded = fread(fileID,'float32');
fclose(fileID);

fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights.bin']);
weights_t_loaded = fread(fileID,'float32');
fclose(fileID);

fileID = fopen(['../output/' experimentName '/Neurons_NetworkDelays.bin']);
delays_loaded = fread(fileID,'int32');
fclose(fileID);

fileID = fopen(['../output/' experimentName '/Neurons_NetworkPre.bin']);
preIDs_loaded = fread(fileID,'int32');
fclose(fileID);

fileID = fopen(['../output/' experimentName '/Neurons_NetworkPost.bin']);
postIDs_loaded = fread(fileID,'int32');
fclose(fileID);

cond = find(preIDs_loaded>=0);
preIDs_loaded = preIDs_loaded(cond)+1; %index start from 1 in matlab
postIDs_loaded = postIDs_loaded(cond)+1;
weights_u_loaded = weights_u_loaded(cond);
weights_t_loaded = weights_t_loaded(cond);
delays_loaded = delays_loaded(cond);


%load only FF
cond1 = mod(postIDs_loaded-1,(ExcitDim*ExcitDim+InhibDim*InhibDim))+1<=ExcitDim*ExcitDim;
cond2 = mod(preIDs_loaded-1,(ExcitDim*ExcitDim+InhibDim*InhibDim))+1<=ExcitDim*ExcitDim;
cond3 = postIDs_loaded > preIDs_loaded;
cond4 = floor(postIDs_loaded./(ExcitDim*ExcitDim+InhibDim*InhibDim))~=floor(preIDs_loaded./(ExcitDim*ExcitDim+InhibDim*InhibDim));

FFWeights_u = weights_u_loaded(find(cond1==1 & cond2==1 & cond3==1 & cond4==1));
FFWeights_t = weights_t_loaded(find(cond1==1 & cond2==1 & cond3==1& cond4==1));
FFPreIDs = preIDs_loaded(find(cond1==1 & cond2==1 & cond3==1& cond4==1));
FFPostIDs = postIDs_loaded(find(cond1==1 & cond2==1 & cond3==1& cond4==1));
delays_loaded = delays_loaded(find(cond1==1 & cond2==1 & cond3==1& cond4==1)).*timestep*1000;


tmp_u = abs(FFWeights_u(1:2:length(FFWeights_u))-FFWeights_u(2:2:length(FFWeights_u)));
tmp_t = abs(FFWeights_t(1:2:length(FFWeights_t))-FFWeights_t(2:2:length(FFWeights_t)));
tmp_d = abs(delays_loaded(1:2:length(delays_loaded))-delays_loaded(2:2:length(delays_loaded)));

% tmp_u = FFWeights_u(1:2:length(FFWeights_u))-FFWeights_u(2:2:length(FFWeights_u));
% tmp_t = FFWeights_t(1:2:length(FFWeights_t))-FFWeights_t(2:2:length(FFWeights_t));
% 
% 
% tmp_u = abs(FFWeights_u(1:2:length(FFWeights_u))-FFWeights_u(2:2:length(FFWeights_u))).*max(FFWeights_u(1:2:length(FFWeights_u)), FFWeights_u(2:2:length(FFWeights_u)));
% tmp_t = abs(FFWeights_t(1:2:length(FFWeights_t))-FFWeights_t(2:2:length(FFWeights_t))).*max(FFWeights_t(1:2:length(FFWeights_t)), FFWeights_t(2:2:length(FFWeights_t)));

figure;
untrainedbinHeights = histcounts(tmp_u,'BinWidth',0.025);
trainedbinHeights = histcounts(tmp_t,'BinWidth',0.025);
bar([0.025:0.025:1.0],trainedbinHeights./untrainedbinHeights,'FaceColor',[0.5 0.5 0.5]);
xlim([0, 1.025])
title({'The Factors by which the number of pairs';'with a given weight difference increased during training'})
ylabel({'The factor by which the number of pairs changed';'from the untrained network to the trained network'}')
xlabel('Weight Difference Between Pairs of Synapses')


prob_table_t = zeros(10,10);
prob_table_u = zeros(10,10);
tmp_table_t = zeros(10,10);
tmp_table_u = zeros(10,10);
eps = 0.0001;
for i=1:length(tmp_u)
    bin_d = min(int32(ceil(tmp_d(i)+eps)),10);
    
    bin_t = min(int32(ceil(tmp_t(i)*10+eps)),10);
    bin_u = min(int32(ceil(tmp_u(i)*10+eps)),10);
    
%     if (bin_u==bin_t)
%         continue;
%     end
    
%     [tmp_d(i) tmp_u(i) tmp_t(i) ]
%     [bin_d bin_u bin_t]
    tmp_table_t(bin_d,bin_t)=tmp_table_t(bin_d,bin_t)+1;
    tmp_table_u(bin_d,bin_u)=tmp_table_u(bin_d,bin_u)+1;
    
end
prob_table_t = tmp_table_t;
prob_table_u = tmp_table_u;
% for i=1:10
%     prob_table_t(i,:) = tmp_table_t(i,:)./sum(tmp_table_t(i,:));
%     prob_table_u(i,:) = tmp_table_u(i,:)./sum(tmp_table_u(i,:));
%     
%     
%     prob_table_t(:,i) = tmp_table_t(:,i)./sum(tmp_table_t(:,i));
%     prob_table_u(:,i) = tmp_table_u(:,i)./sum(tmp_table_u(:,i));
% end



% subplot(1,3,1)
% imagesc(prob_table_u)
% title('Untrained');
% ylabel('delay difference');
% xlabel('weight difference');
% set(gca,'XTickLabel',{'0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'})
% set(gca,'YTickLabel',{'0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10'})
% 
% subplot(1,3,2)
% imagesc(prob_table_t)
% title('Trained');
% ylabel('delay difference');
% xlabel('weight difference');
% set(gca,'XTickLabel',{'0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'})
% set(gca,'YTickLabel',{'0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10'})
% 
% subplot(1,3,3)
% imagesc(prob_table_t./prob_table_u)
% title('Trained / Untrained');
% ylabel('delay difference');
% xlabel('weight difference');
% set(gca,'XTickLabel',{'0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0'})
% set(gca,'YTickLabel',{'0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10'})
% 
% 
% figure
% subplot(1,2,1);
% % plot(tmp_d(tmp_u==1 | tmp_u==0),tmp_u(tmp_u==0 | tmp_u==1),'.')
% plot(tmp_d,tmp_u,'.')
% ylim([-0.1 1.1]);
% xlim([-1 11]);
% title('untrained');
% xlabel('Delay differences')
% ylabel('Synaptic weight differences')
% 
% subplot(1,2,2);
% % plot(tmp_d(tmp_t==0 | tmp_t==1),tmp_t(tmp_t==0 | tmp_t==1),'.')
% plot(tmp_d,tmp_t,'.')
% ylim([-0.1 1.1]);
% xlim([-1 11]);
% title('trained');
% xlabel('Delay differences')
% 
% 
% h1 = histogram(tmp_u,'FaceColor','w');
% hold on
% h2 = histogram(tmp_t,'FaceColor','k');
% h1.BinWidth=0.01;
% h2.BinWidth=0.01;
% xlim([0.5 1.0]);
% ylim([0 5000]);
% 
% title('Absolute Difference between Weights at each Synapse');
% ylabel('Number of Synapses')
% xlabel('Absolute Difference between Weights at each Synapse')

% figindex = 0;
% for i = 1:length(FFWeights_u/2)
%     if abs(FFWeights_t(i*2-1)-FFWeights_t(i*2))>0.9 && abs(FFWeights_t(i*2-1)-FFWeights_u(i*2-1))>0.3
%         figindex=figindex+1;
%         subplot(3,3,figindex);
%         plot([FFWeights_u(i*2-1) FFWeights_t(i*2-1)],'k', 'LineWidth', 2);
%         hold on;
%         plot([FFWeights_u(i*2) FFWeights_t(i*2)],'k--','LineWidth', 2);
%         set(gca,'Xtick',1:2,'XTickLabel',{'untrained','trained'})
%         ylabel('Synaptic weight');
%         title(['j=' num2str(FFPreIDs(i*2-1)) ', i=' num2str(FFPostIDs(i*2-1)) ' d1 = ' num2str(delays_loaded(i*2-1)) ' d2=' num2str(delays_loaded(i*2))]);
%     end
% 
% end




