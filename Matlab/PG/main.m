experimentName = '1.4--FF_FB_LAT_stdp_0.005';
plotOn = 0;
%targetNeuronIDs = [12346,17525]%star
% targetNeuronIDs = [13581,19163]%heart
targetNeuronIDs = [12686,18657]%circle


tmp = double(reshape(inputImg(:,:,1),128,128))/256;

subplotIndex = 1;
for targetID = targetNeuronIDs
    subplot(2,length(targetNeuronIDs),subplotIndex);
    weightMap = traceGabor(experimentName, targetID, plotOn);
    colormap(gray);
    imagesc(1-transpose(weightMap),[0,1]);
    title(['cell id:' num2str(targetID)]);
    
    subplot(2,length(targetNeuronIDs),length(targetNeuronIDs)+subplotIndex);
    imagesc(tmp+(transpose(weightMap)),[0,1])
    subplotIndex=subplotIndex+1;
end

% subplot(2,length(targetNeuronIDs),length(targetNeuronIDs)+1)
% imagesc(cdata);


imagesc(tmp+(transpose(weightMap)),[0,1])