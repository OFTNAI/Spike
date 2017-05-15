
experimentName = '1.4--FF_FB_LAT_stdp_0.005'
timestep = 0.00002;
presentationTime = 2.0;
nObjs = 3;
nTrans = 2;
ExcitDim = 64;
InhibDim = 32;
nLayers = 4;

plotRange = [0.5 0.6];
precision = 300;

for trainedNet = [1]
    if(trainedNet)
%         load(['../output/' experimentName '/groups_trained.mat']);
        fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Trained_Epoch0.bin']);
        fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Trained_Epoch0.bin']);
    else
%         load(['../output/' experimentName '/groups_untrained.mat']);
        fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Untrained_Epoch0.bin']);
        fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Untrained_Epoch0.bin']);
    end
    
    spikes_id = fread(fileID_id,'int32');
    fclose(fileID_id);
    
    spikes_time = fread(fileID_time,'float32');%second
    fclose(fileID_time);
    
    
% %     selectedCells = [10396,15455,15580];
% %     selectedCells = [16130,16131,10954];
%     selectedCells = [16068,16137,10561];
% 
%     cond = ismember(spikes_id,selectedCells);
%     id_subset = spikes_id(cond);
%     time_subset = spikes_time(cond);
%     
    figure;
%     plot(time_subset,id_subset,'.')
% %     plot(spikes_time,spikes_id,'.')
%     xlim([0,nObjs*nTrans*presentationTime]);
%     ylim([(ExcitDim*ExcitDim+InhibDim*InhibDim)*2,(ExcitDim*ExcitDim+InhibDim*InhibDim)*4])
%     
%     ['test'];
    

    for obj=0:nObjs-1
        for trans=0:nTrans-1 
    %                 disp([trigger_index, obj, trans]);
            time_begin = (obj*nTrans+trans)*presentationTime;
            time_end = time_begin + presentationTime;
            cond1 = (spikes_time>time_begin) & (spikes_time<=time_end);
            
            for l=1:nLayers
                cond2_excit = spikes_id>(ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1) & spikes_id<=(ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1)+(ExcitDim*ExcitDim);
                cond2_inhib = spikes_id>(ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1)+(ExcitDim*ExcitDim) & spikes_id<=(ExcitDim*ExcitDim+InhibDim*InhibDim)*l;
                               
%                 [(l-1)*nObjs*nTrans+obj*nTrans+trans+1]
                
%                 subplot(nLayers,nObjs*nTrans,(nLayers-l)*nObjs*nTrans+obj*nTrans+trans+1);
                [((nLayers-l)*nObjs*nTrans*3)+(obj*nTrans+trans+1)+0*(nObjs*nTrans) ((nLayers-l)*nObjs*nTrans*3)+(obj*nTrans+trans+1)+1*(nObjs*nTrans)]
                subplot(nLayers*3,nObjs*nTrans,[((nLayers-l)*nObjs*nTrans*3)+(obj*nTrans+trans+1)+0*(nObjs*nTrans) ((nLayers-l)*nObjs*nTrans*3)+(obj*nTrans+trans+1)+1*(nObjs*nTrans)]);
                
                id_subset_excit = spikes_id(cond1 & cond2_excit);
                time_subset_excit = spikes_time(cond1 & cond2_excit);
                
                
                
                
                %plot raster
                plot(time_subset_excit,id_subset_excit,'.')
%                 xlim([time_begin,time_end]);
                xlim([time_begin+presentationTime*plotRange(1),time_begin+presentationTime*plotRange(2)]);
                ylim([(ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1),(ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1)+(ExcitDim*ExcitDim)])
%                 set(gca,'xtick',[])
                set(gca,'xaxisLocation','top')
                set(gca,'ytick',[])
                
                
                
                
                
                %plot histgram
                histTable = zeros(1,presentationTime*precision);
                for i = 1:length(time_subset_excit)
                    histTable(1,int32(ceil((time_subset_excit(i)-time_begin)*precision)))=histTable(1,int32(ceil((time_subset_excit(i)-time_begin)*precision)))+1;
                end
                
                subplot(nLayers*3,nObjs*nTrans,[((nLayers-l)*nObjs*nTrans*3)+(obj*nTrans+trans+1)+2*(nObjs*nTrans)]);
                bar(histTable);
%                 xlim([1+1*precision,length(histTable)-0.5*precision]);
                xlim([1+(presentationTime*plotRange(1))*precision,1+(presentationTime*plotRange(2)*precision)]);

%                 ylim([0 20]);
                set(gca,'xtick',[])
                
                
%                 id_subset_inhib = spikes_id(cond1 & cond2_inhib);
%                 time_subset_inhib = spikes_time(cond1 & cond2_inhib);
                
                
            end
        end
    end
    

    
end