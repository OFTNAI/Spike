function FR = tmpScriptPlotFROfBindingCells(id,lineSpec)
    
    nExcitCells = 64*64;
    nInhibCells = 32*32;
    experimentName = '6.3grayAndBlack_a--StatDec_FFLATFB_20EP_FB5_rand123';
    FRThreshold = 100.0;
    
    layer = floor((id-1.0)/(nExcitCells+nInhibCells))+1
    cellId = mod(id,nExcitCells+nInhibCells)
    
    %%load untrained FR
    trainedNet = 0;
    FR_u = loadFR(experimentName,layer,trainedNet);
    FR_u(FR_u>FRThreshold) = FRThreshold;

    trainedNet = 1;
    FR_t = loadFR(experimentName,layer,trainedNet);
    FR_t(FR_t>FRThreshold) = FRThreshold;
    
    plot([FR_t(1,:,cellId) FR_t(2,:,cellId)],lineSpec,'LineWidth',2);
    xlim([0.8 8.2])
    ylim([-5 105])
%     title('cell id: 1726')