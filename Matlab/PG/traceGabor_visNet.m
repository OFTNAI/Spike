% traceGabor.m
% VisBack
%
% Created by Akihiro Eguchi on 23/04/15.
% Copyright 2011 OFTNAI. All rights reserved.
%


function weightMap = traceGabor(experimentName, targetLayer,targetRow,targetCol, plotOn)
    % Import global variables
%     declareGlobalVars();

%     networkFile = 'TrainedNetwork.txt';
    
    
    
    
    fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights.bin']);
    weights_loaded = fread(fileID,'float32');
    fclose(fileID);

    fileID = fopen(['../output/' experimentName '/Neurons_NetworkPre.bin']);
    preIDs_loaded = fread(fileID,'int32');
    fclose(fileID);

    fileID = fopen(['../output/' experimentName '/Neurons_NetworkPost.bin']);
    postIDs_loaded = fread(fileID,'int32');
    fclose(fileID);
    
    
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Invariance Plots
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    v1Dimension = 128;
    weightMap = zeros(v1Dimension,v1Dimension);

    % Open file
%     connectivityFileID = fopen([folder '/' networkFile]);

    % Read header
%     [networkDimensions, neuronOffsets2] = loadWeightFileHeader(connectivityFileID);
%     v1Dimension = networkDimensions(1).dimension;

    % Setup vars
%     numRegions = length(networkDimensions);
    numRegions = 4;
    depth = 1;
    
    Threshold = [0.6, 0.9, 0.9, 0.95]; %threshold sets the persentage of the synaptic connections traced from each cell.

    %%%feature plotting parameters%%%
    %Phases = [0, 180, -90, 90];%org.
    Phases = [0, 180];
    Orientations = [0, 45, 90, 135];%org.
    %Wavelengths = [2, 16];
    %Wavelengths = [2, 4, 8, 16];
    Wavelengths = [2];


    if targetLayer>numRegions
        ['target layer is too big'];
        return
    end

    features = findV1Sources(targetLayer, depth, 1, targetRow, targetCol);
    numFeatures = length(features);

    if numFeatures > 0
        for k = 1:numFeatures
            drawFeature(features(k).row, features(k).col, features(k).depth, features(k).weight);
        end
        weightMap = weightMap./(max(weightMap(:))+0.0001);
        if(plotOn)
            fig = figure;
            colormap(gray);
            imagesc(1-weightMap,[0,1]);
            freezeColors 
            colormap('default');
        end
%         saveas(fig,[folder '/l' num2str(targetLayer) '_row' num2str(targetRow) '_col' num2str(targetCol) '.fig']);
%         saveas(fig,[folder '/l' num2str(targetLayer) '_row' num2str(targetRow) '_col' num2str(targetCol) '.png']);

    else
        plot([(v1Dimension+1)/2 (v1Dimension+1)/2], [0 v1Dimension+1], 'r');
        hold on;
        plot([0 v1Dimension+1], [(v1Dimension+1)/2 (v1Dimension+1)/2], 'r');
    end
    


    function [synapses] = afferentSynapseList(neuronOffsets, layer, depth, row, col)
        % Import global variables
%         global SOURCE_PLATFORM_USHORT;
%         global SOURCE_PLATFORM_FLOAT;

        % Find offset of synapse list of neuron region.(depth,i,j)
        %fseek(fileID, neuronOffsets{region}(row, col, depth).offset, 'bof');
        
%         experimentName
        
        synID = layer*(row*


        % Allocate synapse struct array
        afferentSynapseCount = neuronOffsets{layer}(row, col, depth).afferentSynapseCount;
        synapses(afferentSynapseCount).layer = [];
        synapses(afferentSynapseCount).depth = [];
        synapses(afferentSynapseCount).row = [];
        synapses(afferentSynapseCount).col = [];
        synapses(afferentSynapseCount).weight = [];

        % Fill synapses
        for s = 1:afferentSynapseCount,
            v = fread(fileID, 4, SOURCE_PLATFORM_USHORT);

            synapses(s).layer = v(1)+1;
            synapses(s).depth = v(2)+1;
            synapses(s).row = v(3)+1;
            synapses(s).col = v(4)+1;
            synapses(s).weight = fread(fileID, 1, SOURCE_PLATFORM_FLOAT);
        end
    end



    function [sources] = findV1Sources(layer, depth, weight, row, col)

        if layer == 1, % termination condition, V1 cells return them self
            % Make 1x1 struct array
            sources(1).layer = layer;
            sources(1).row = row;
            sources(1).col = col;
            sources(1).depth = depth;
            sources(1).weight = weight;

        elseif layer > 1,

            synapses = afferentSynapseList(neuronOffsets2, layer, depth, row, col);

            sources = [];

            wTmp = zeros(1,length(synapses));
            for s=1:length(synapses) % For each child
                wTmp(1,s)= synapses(s).weight;
            end
            
            %rand order
            randIndex = randperm(length(synapses));
            
            w = [randIndex; wTmp(1,randIndex)];
            
            sorted = transpose(sortrows(transpose(w),2));
            sortedIndex = sorted(1,length(synapses):-1:round(length(synapses)*Threshold(1,targetLayer))); %take top x % of the synapses

            for s=sortedIndex%1:length(synapses) % For each child
                if synapses(s).region<layer
                    sources = [sources findV1Sources(synapses(s).layer, synapses(s).depth, (synapses(s).weight*weight), synapses(s).row, synapses(s).col)];
                end
            end
        end
    end


    function drawFeature(row, col, depth, weight)
        halfSegmentLength = 2;%0.5;
        [orientation, wavelength, phase] = decodeDepth(depth);

        featureOrrientation = -1*orientation + 90; % orrientation is the param to the filter, but it corresponds to a perpendicular image feature

        dx = cos(deg2rad(featureOrrientation));
        dy = sin(deg2rad(featureOrrientation));
        %[featureOrrientation dx dy]

        %disp([depth]);
        %disp([featureOrrientation]);


    %         x1 = col - dx;
    %         x2 = col + dx;
    %         y1 = row - dy;
    %         y2 = row + dy;

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


    %         dx = halfSegmentLength*cos(deg2rad(featureOrrientation))*weight;
    %         dy = halfSegmentLength*sin(deg2rad(featureOrrientation))*weight;
    %         x1 = col - dx;
    %         x2 = col + dx;
    %         y1 = row - dy;
    %         y2 = row + dy;


        %plot([x1 x2], [y1 y2], '-r', 'Color', [1-weight 1-weight 1-weight], 'LineWidth', 1.2*weight);
        %hold on;

    end

    function [orientation, wavelength, phase] = decodeDepth(depth)
        %disp([depth]);
    %         
    %         depth = uint8(depth)-1; % These formula expect C indexed depth, since I copied from project
    %         disp([depth]);
    %         
    %         w = mod((idivide(depth, length(Phases))), length(Wavelengths));
    %         wavelength = Wavelengths(w+1);
    %         
    %         ph = mod(depth, length(Phases));
    %         phase = Phases(ph+1);
    %         
    %         o = idivide(depth, (length(Wavelengths) * length(Phases)));
    %         
    %         %disp([o]);
    %         orrientation = Orrientations(o+1);



        %depth = uint8(depth)-1; % These formula expect C indexed depth, since I copied from project
        %disp([depth]);

        depth = uint8(depth);
        if(depth>length(Wavelengths)*length(Phases)*length(Orientations))
            disp(['ERROR: Check the parameters for the filtering at line 58']);
            return
        end
    %         or_index = mod(depth-1,length(Orrientations))+1;
    %         ph_index = mod(idivide(depth-1,length(Orrientations)),length(Phases))+1;
    %         wave_index = idivide(depth-1,length(Orrientations)*length(Phases))+1;

        % x = Phase, y = wavelength, z = orientation
        %*ilustration of indexing of depth
        %orientation = 1      orientation = 2
        %    phase                phase
        %  w  1  2  3           w  7  8  9
        %  a  4  5  6           a 10 11 12
        %  v  1  2  3           v 13 14 15
        %  e  4  5  6           e 16 17 18
        ph_index = mod(depth-1,length(Phases))+1;
        wave_index = mod(idivide(depth-1,length(Phases)),length(Wavelengths))+1;
        or_index = idivide(depth-1,length(Wavelengths)*length(Phases))+1;

        %[depth or_index ph_index wave_index]
        phase = Phases(ph_index);
        wavelength = Wavelengths(wave_index);
        orientation = Orientations(or_index);
    end  



end