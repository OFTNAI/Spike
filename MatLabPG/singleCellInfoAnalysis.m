experimentName = '1.4--FF_FB_LAT_stdp_0.005';
layer = 4;

%for trainedNet = [0 1]
for trainedNet = [1]
    FR = loadFR(experimentName,layer,trainedNet);


    %info analysis

    FRThreshold = 100.0;
    nBins = 5;
    FR(FR>FRThreshold) = FRThreshold;

    IRs = infoAnalysis(FR,trainedNet,nBins);
    
    fig=figure;
    
    for obj = 1:3
        infoPerObj = IRs(:,obj);
        
        [B, I] = sort(-1*infoPerObj);
        B = B*-1;
        numMaxInfo = length(find(B>=1.58));
        
        indexOfCellsWithMaxInfo = I(1:numMaxInfo) + (64*64+32*32)*(layer-1);
        
        plotCount = 0;
        subplotWidth = 3;
        for cellIndex = 1:numMaxInfo
            subplotIndex = mod(cellIndex-1,subplotWidth*subplotWidth)+1;
            subplot(subplotWidth,subplotWidth,subplotIndex);
            
            targetID = indexOfCellsWithMaxInfo(cellIndex);
            weightMap = traceGabor(experimentName, targetID, plotOn);
            colormap(gray);
            imagesc(1-weightMap,[0,1]);
            
            cellIndexWithinLayer = mod(targetID-1,64*64+32*32)+1;
            l = ceil(targetID/(64*64+32*32));
            y = mod(cellIndexWithinLayer-1,64)+1;
            x = ceil(cellIndexWithinLayer/64);
            
            title(['id:' num2str(targetID) ' l:' num2str(l) ' y:' num2str(y) ' x:' num2str(x)]);
            
            if(subplotIndex==subplotWidth*subplotWidth)
                %save;
                suptitle(['obj:' num2str(obj)]);
                'save';
                saveas(fig,['../output/' experimentName '/' num2str(trainedNet) '_gaborTrace_obj_' num2str(obj) '_' num2str(plotCount) '.fig']);
                set(gcf,'PaperPositionMode','auto')
                print(['../output/' experimentName '/' num2str(trainedNet) '_gaborTrace_obj_' num2str(obj) '_' num2str(plotCount)],'-dpng','-r0');
                clf;
                plotCount=plotCount+1;
            end
        end
        
        if(subplotIndex~=subplotWidth*subplotWidth)
            %save;
            suptitle(['obj:' num2str(obj)]);
            saveas(fig,['../output/' experimentName '/' num2str(trainedNet) '_gaborTrace_obj_' num2str(obj) '_' num2str(plotCount) '.fig']);
            set(gcf,'PaperPositionMode','auto')
            print(['../output/' experimentName '/' num2str(trainedNet) '_gaborTrace_obj_' num2str(obj) '_' num2str(plotCount)],'-dpng','-r0');
            'save';
            clf;
        end
        
        
    end
    
    
    
end