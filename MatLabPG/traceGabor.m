% traceGabor.m
% VisBack
%
% Created by Akihiro Eguchi on 23/04/15.
% Copyright 2011 OFTNAI. All rights reserved.
%




function weightMap = traceGabor(experimentName, targetNeuronID, plotOn)

    traceMode = 2; %0: trace by threshold, 1: trace by fixed number of connections, 2:take min(threshold,fixed number)
    weightTh = 0.99;
    v1Dimension = 128;
    numberOfConTracePerCell = 15;

    ExcitDim = 64;
    InhibDim = 32;
%     nLayers = 4;
%     targetLayer = floor(targetNeuronID/(ExcitDim*ExcitDim+InhibDim*InhibDim));
    
    %load network
    fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights.bin']);
    weights_loaded = fread(fileID,'float32');
    fclose(fileID);

    fileID = fopen(['../output/' experimentName '/Neurons_NetworkPre.bin']);
    preIDs_loaded = fread(fileID,'int32');
    fclose(fileID);

    fileID = fopen(['../output/' experimentName '/Neurons_NetworkPost.bin']);
    postIDs_loaded = fread(fileID,'int32');
    fclose(fileID);
    

    cond1 = mod(postIDs_loaded-1,(ExcitDim*ExcitDim+InhibDim*InhibDim))+1<=ExcitDim*ExcitDim;% post id is within ex layer
    cond2 = ((mod(preIDs_loaded-1,(ExcitDim*ExcitDim+InhibDim*InhibDim))+1<=ExcitDim*ExcitDim)|preIDs_loaded<0);% pre id is within ex layer
    cond3 = ((floor(postIDs_loaded./(ExcitDim*ExcitDim+InhibDim*InhibDim))-floor(preIDs_loaded./(ExcitDim*ExcitDim+InhibDim*InhibDim))==1)|preIDs_loaded<0);%feed forward con

    FFWeights = weights_loaded(cond1 & cond2 & cond3);
    FFPreIDs = preIDs_loaded(cond1 & cond2 & cond3);
    FFPostIDs = postIDs_loaded(cond1 & cond2 & cond3);
    
    Phases = [0, 180];
    Orientations = [0, 45, 90, 135];
    Wavelengths = [2];
    
    weightMap = zeros(v1Dimension,v1Dimension);%To-Do should be v1Dimension*v1Dimension map
    
    
    features = findV1Sources(targetNeuronID, 1);
    
    
    numFeatures = length(features);
    count = 0;
    if numFeatures > 0
        count = count+1;
        disp(num2str(count));
        for k = 1:numFeatures
            drawFeature(features(k).id, features(k).weight);
        end
        weightMap = weightMap./(max(weightMap(:))+0.0001);
%         if(plotOn)
%             fig = figure;
%             colormap(gray);
%             imagesc(1-weightMap,[0,1]);
%             freezeColors 
%             colormap('default');
%         end
%         saveas(fig,[folder '/l' num2str(targetLayer) '_row' num2str(targetRow) '_col' num2str(targetCol) '.fig']);
%         saveas(fig,[folder '/l' num2str(targetLayer) '_row' num2str(targetRow) '_col' num2str(targetCol) '.png']);

    else
        plot([(v1Dimension+1)/2 (v1Dimension+1)/2], [0 v1Dimension+1], 'r');
        hold on;
        plot([0 v1Dimension+1], [(v1Dimension+1)/2 (v1Dimension+1)/2], 'r');
    end
    
    
  
    

    function [features] = findV1Sources(targetNeuronID, weight)
        if targetNeuronID < 0 % termination condition, V1 cells return them self
            % Make 1x1 struct array
            features(1).id = targetNeuronID;
            features(1).weight = weight;

        else
            
            
            if(traceMode==0)
% 
                preIDsFromTarget = FFPreIDs(FFPostIDs==targetNeuronID & (FFWeights>weightTh|FFPreIDs<0));
                weightFromTarget = FFWeights(FFPostIDs==targetNeuronID & (FFWeights>weightTh|FFPreIDs<0));


%                 preIDsFromTarget = FFPreIDs(FFPostIDs==targetNeuronID & FFWeights>weightTh);
%                 weightFromTarget = FFWeights(FFPostIDs==targetNeuronID & FFWeights>weightTh);

                features = [];

                for s=1:length(weightFromTarget)
                    features = [features findV1Sources(preIDsFromTarget(s), weightFromTarget(s)*weight)];
                end
                
            elseif(traceMode == 1)
                preIDsFromTarget = FFPreIDs(FFPostIDs==targetNeuronID);
                weightFromTarget = FFWeights(FFPostIDs==targetNeuronID);

                [B, I] = sort(-1*weightFromTarget);

                features = [];

                for s=1:numberOfConTracePerCell
                    features = [features findV1Sources(preIDsFromTarget(I(s)), weightFromTarget(I(s))*weight)];
                end
            elseif(traceMode ==2)
                preIDsFromTarget = FFPreIDs(FFPostIDs==targetNeuronID & FFWeights>weightTh);
                weightFromTarget = FFWeights(FFPostIDs==targetNeuronID & FFWeights>weightTh);

                [B, I] = sort(-1*weightFromTarget);

                features = [];

%                 length(weightFromTarget)
                for s=1:min(numberOfConTracePerCell,length(weightFromTarget))
                    features = [features findV1Sources(preIDsFromTarget(I(s)), weightFromTarget(I(s))*weight)];
                end
            end
        end
    end



    function drawFeature(targetNeuronID, weight)
        
        inputID = -1*targetNeuronID;
        filterID = uint8(ceil(inputID/(v1Dimension*v1Dimension)));
        row = ceil((mod(inputID-1,v1Dimension*v1Dimension)+1)/v1Dimension);
        col = mod((mod(inputID-1,v1Dimension*v1Dimension)),v1Dimension)+1;
        
      
        weightMap(row,col)=weightMap(row,col)+weight;
        
        ph_index = mod(filterID-1,length(Phases))+1;
        wave_index = mod(idivide(filterID-1,length(Phases)),length(Wavelengths))+1;
        or_index = idivide(filterID-1,length(Wavelengths)*length(Phases))+1;

        %[depth or_index ph_index wave_index]
        phase = Phases(ph_index);
        wavelength = Wavelengths(wave_index);
        orientation = Orientations(or_index);
        
        
        halfSegmentLength = 2;%0.5;
        
        featureOrientation = -1*orientation + 90; % orrientation is the param to the filter, but it corresponds to a perpendicular image feature

        dx = cos(deg2rad(featureOrientation));
        dy = sin(deg2rad(featureOrientation));

        for change = 0:halfSegmentLength
            if change == 0
                x1 = col;
                y1 = row;
                %[x1 y1]
                weightMap(y1,x1) = weightMap(y1,x1)+weight;
            else
                x1 = mod(round(col - change * dx - 1),v1Dimension)+1;
                y1 = mod(round(row - change * dy - 1),v1Dimension)+1;
                %[x1 y1]
                weightMap(y1,x1) = weightMap(y1,x1)+weight;

                x1 = mod(round(col + change * dx -1),v1Dimension)+1;
                y1 = mod(round(row + change * dy -1),v1Dimension)+1;
                weightMap(y1,x1) = weightMap(y1,x1)+weight;
            end
        end


    end
  


end