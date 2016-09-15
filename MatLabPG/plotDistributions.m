function plotDistributions(ppre,nLayers,excitLayerDim,inhibLayerDim)


% nLayers = 4;
% excitLayerDim = 64;
% inhibLayerDim = 32;
nCellsInExcitLayer = excitLayerDim*excitLayerDim;
nCellsInInhibLayer = inhibLayerDim*inhibLayerDim;
figure;
for l=1:nLayers
    subplot(nLayers,2,(nLayers-l+1)*2-1);
    %plot pre of 1st inhib layer
    postIndex = nCellsInInhibLayer*(l-1) + nCellsInExcitLayer*l + nCellsInInhibLayer/2 + 1;
    inputIndex = ppre{postIndex} - (nCellsInExcitLayer+nCellsInInhibLayer)*(l-1);
    inputIndex = inputIndex(find(inputIndex > 0));
    rows = ceil(inputIndex/excitLayerDim);
    cols = mod(inputIndex-1,excitLayerDim)+1;
    plot(rows,cols,'ro');
    xlim([1 excitLayerDim])
    ylim([1 excitLayerDim])
    hold on;
    
    subplot(nLayers,2,(nLayers-l+1)*2);
    postTmp = 1;
    plot(ceil(postTmp/inhibLayerDim), mod(postTmp-1,inhibLayerDim)+1,'r*');
    xlim([1 inhibLayerDim])
    ylim([1 inhibLayerDim])
    hold on;
    
    %plot pre of 1st excit layer 
    postIndex = (nCellsInInhibLayer+nCellsInExcitLayer)*(l-1)  + excitLayerDim;
    inputIndex = ppre{postIndex} - (nCellsInExcitLayer)*(l) - nCellsInInhibLayer*(l-1);
    inputIndex = inputIndex(find(inputIndex > 0));
    rows = ceil(inputIndex/inhibLayerDim);
    cols = mod(inputIndex-1,inhibLayerDim)+1;
    plot(rows,cols,'go');
    xlim([1 inhibLayerDim])
    ylim([1 inhibLayerDim])
    hold on;
    
    subplot(nLayers,2,(nLayers-l+1)*2-1);
    postTmp = excitLayerDim;
    plot(ceil(postTmp/excitLayerDim), mod(postTmp-1,32)+1,'g*');
    hold on;
    
    %plot FF
    if(l>1)
        subplot(nLayers,2,(nLayers-l+2)*2-1);
        postIndex = (nCellsInInhibLayer+nCellsInExcitLayer)*(l-1)  + nCellsInExcitLayer;
        inputIndex = ppre{postIndex} - (nCellsInExcitLayer+nCellsInInhibLayer)*(l-2);
        inputIndex = inputIndex(find(nCellsInExcitLayer>=inputIndex & inputIndex > 0));
        rows = ceil(inputIndex/excitLayerDim);
        cols = mod(inputIndex-1,excitLayerDim)+1;
        plot(rows,cols,'bo');
        xlim([1 excitLayerDim])
        ylim([1 excitLayerDim])
        hold on;

        subplot(nLayers,2,(nLayers-l+1)*2-1);
        subplot(nLayers,2,(nLayers-l+1)*2-1);
        postTmp = nCellsInExcitLayer;
        plot(ceil(postTmp/excitLayerDim), mod(postTmp-1,excitLayerDim)+1,'b*');
        hold on;
    end
    
    
    
    
    
%     postIndex = (32*32+16*16)*1+32;
%     inputIndex = ppre{postIndex};
%     inputIndex = inputIndex(find(inputIndex<(32*32+16*16)));
%     rows = ceil(inputIndex/32);
%     cols = mod(inputIndex-1,32)+1;
%     plot(rows,cols,'o');
%     xlim([1 32])
%     ylim([1 32])
end