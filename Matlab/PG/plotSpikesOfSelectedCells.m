
experimentName = '1.4--FF_FB_LAT_stdp_0.005'
timestep = 0.00002;
presentationTime = 2.0;
nObjs = 3;
nTrans = 2;



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
    
    
%     selectedCells = [10396,15455,15580];
%     selectedCells = [16130,16131,10954];
    selectedCells = [16068,16137,10561];

    cond = ismember(spikes_id,selectedCells);
    id_subset = spikes_id(cond);
    time_subset = spikes_time(cond);
    
    figure;
    plot(time_subset,id_subset,'.')
%     plot(spikes_time,spikes_id,'.')
    xlim([0,nObjs*nTrans*presentationTime]);
    ylim([(64*64+32*32)*2,(64*64+32*32)*4])
    
    ['test'];
    
%     triggerCountMatrix = zeros(nObjs,nTrans,length(groups));
%     
%     for trigger_index = 1:length(groups)
% %        firings = groups{1,trigger_id}.firings;
%         index_t = 1;
%         trigger_ids = groups{1,trigger_index}.firings(1:anchorWidth,2);
%         trigger_times = (groups{1,trigger_index}.firings(1:anchorWidth,1)-1)*timestep;
% %         disp(round(trigger_index*100.0/length(groups)));
%         for obj=0:nObjs-1
%             for trans=0:nTrans-1
% %                 disp([trigger_index, obj, trans]);
%                 time_benig = (obj*nTrans+trans)*presentationTime;
%                 time_end = time_benig + presentationTime;
%                 cond = (spikes_time>=time_benig) & (spikes_time<time_end);
%                 id_subset = spikes_id(cond);
%                 time_subset = spikes_time(cond);
%                 
%                 for i=1:length(id_subset)
%                     if id_subset(i)==trigger_ids(1)
%                         flag = true;
%                         for a=2:anchorWidth
%                             timimg = time_subset(i)+trigger_times(a);
%                             cond2 = (id_subset==trigger_ids(a)) & (abs(time_subset-timimg)<jitterSize);
%                             if(length(id_subset(cond2))==0)
%                                 flag = false;
%                                 break;
%                             end
%                         end
%                         if(flag)
%                             triggerCountMatrix(obj+1,trans+1,trigger_index) = triggerCountMatrix(obj+1,trans+1,trigger_index)+1;
%                             disp(['Found! -- trigger:' num2str(trigger_index) ' obj:' num2str(obj) ' trans:' num2str(trans)]);
%                         end
%                     end
%                 end
% 
%                 
%                 id_subset;
%                 
%                 
%                 
% %                 while(spikes_time(index_t)<time_end)
% %                     if(spikes_id(index_t)==trigger_ids(1))
% %                         t_init = spikes_time(index_t);
% %                          
% %                     end
% %                     index_t=index_t+1;
% %                 end
% 
%                 
%             end
%         end
%         
%         
%         
%     end
    

    
end