%
% plotFilters.m
% VisBack
%
% Created by Bedeho Mender on 29/04/11.
% Copyright 2011 OFTNAI. All rights reserved.
%
% Input=========
% folder: folder with filtered outputs
% Output========
% 

function plotFilters(folders, V1_size, vOrients, vPhases, vScales)

    %V1_size = 128
    
    %vOrients = [0,45,90,135]
    %vPhases = [0] % ,180
    %vScales = [4,8] %,16
    
    %vPhases = [0]
    %vScales = [4,8]
    %vOrients = [0,45,90,135]
    
    [pathstr, name, ext] = fileparts(folders);
    
    fig = figure();
    
    for o=1:length(vOrients),
        for p=1:length(vPhases),
            for s=1:length(vScales),
                
                ji_temp = [pathstr '/' name '.flt/' name '.' num2str(vScales(s)) '.' num2str(vOrients(o)) '.' num2str(vPhases(p)) '.gbo'];
                om = fopen([pathstr '/' name '.flt/' name '.' num2str(vScales(s)) '.' num2str(vOrients(o)) '.' num2str(vPhases(p)) '.gbo']);    
                v = fread(om, V1_size*V1_size, 'float');
                fclose(om);
                
                meanVal = mean(v);
                stdVal = std(v);
                maxVal = max(v);
                minVal = min(v);
                
                index = length(vPhases)*length(vScales)*(o - 1) + length(vScales)*(p - 1) + s;
                
                pMatrix = reshape(v, [V1_size V1_size]); 
                
                % Read data into matrix, but data is reshaped
                % column wise, while data is saved row wise, sow 
                % we must transpose
                pMatrix = pMatrix'; 
                
                %pMatrix = arrayfun(@ramp, pMatrix);
                
                subplot(length(vOrients), length(vPhases)*length(vScales), index);
                imagesc(pMatrix); 
                
                colorbar
                title({['Orrient ' num2str(vOrients(o)) ', Phase ' num2str(vPhases(p)) ', Scale ' num2str(vScales(s))] ..., 
                        ['Mean ' num2str(meanVal) ', Std ' num2str(stdVal) ', Max ' num2str(maxVal) ', Min ' num2str(minVal)]});
                    
                axis square;
            end
        end
    end
    
    makeFigureFullScreen(fig);
    
end

function r = ramp(x)

    if x < 0,
        r = 0;
    else
        r = x;
    end
end

%function s = sigmoid(x)