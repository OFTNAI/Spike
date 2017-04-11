function FR = loadFR(experimentName,layer,trainedNet)

    presentationTime = 2.0;
    nObjs = 2;
    nTrans = 4;
%     trainedNet = 1;

    ExcitDim = 64;
    InhibDim = 32;
%     nLayers = 4;
    
    FR = zeros(nObjs,nTrans,ExcitDim*ExcitDim);

    
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

    for obj = 1:nObjs
        for trans = 1:nTrans
            timeBegin = ((obj-1)*nTrans+(trans-1)) * presentationTime;
            timeEnd = timeBegin+ presentationTime;
            ['obj ' num2str(obj) ' trans ' num2str(trans) ' begin:' num2str(timeBegin) ' end:' num2str(timeEnd)]
            cond1 =  timeBegin < spikes_time & spikes_time < timeEnd;
            cond2 = (ExcitDim*ExcitDim+InhibDim*InhibDim)*(layer-1)<spikes_id & spikes_id<=(ExcitDim*ExcitDim)*layer+(InhibDim*InhibDim)*(layer-1);
            id_inRange = spikes_id(cond1&cond2) - (ExcitDim*ExcitDim+InhibDim*InhibDim)*(layer-1);
            for i = 1:length(id_inRange)
                FR(obj,trans,id_inRange(i))=FR(obj,trans,id_inRange(i))+1;
            end
        end
    end

end