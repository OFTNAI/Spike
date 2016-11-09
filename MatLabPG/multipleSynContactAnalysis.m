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


figindex = 0;
for i = 1:length(FFWeights_u/2)
    if abs(FFWeights_t(i*2-1)-FFWeights_t(i*2))>0.9 && abs(FFWeights_t(i*2-1)-FFWeights_u(i*2-1))>0.3
        figindex=figindex+1;
        subplot(3,3,figindex);
        plot([FFWeights_u(i*2-1) FFWeights_t(i*2-1)],'k', 'LineWidth', 2);
        hold on;
        plot([FFWeights_u(i*2) FFWeights_t(i*2)],'k--','LineWidth', 2);
        set(gca,'Xtick',1:2,'XTickLabel',{'untrained','trained'})
        ylabel('Synaptic weight');
        title(['j=' num2str(FFPreIDs(i*2-1)) ', i=' num2str(FFPostIDs(i*2-1)) ' d1 = ' num2str(delays_loaded(i*2-1)) ' d2=' num2str(delays_loaded(i*2))]);
    end

end




