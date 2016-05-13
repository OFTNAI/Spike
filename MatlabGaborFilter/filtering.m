function [dim,FI,GF] = filtering(file, psi, scale, orient, bw, gamma, set, paddingColor)
    close all;

    if nargin < 8
        error('Not enough arguments passed');
    end

    finfo = imfinfo(file);
    fsize = finfo.Height * finfo.Width;
    originalImageD = [finfo.Height, finfo.Width];
    paddedImageD = 2 * originalImageD;

    ss = length(scale);
    os = length(orient);
    ps = length(psi);
    FI = cell(ss,os,ps);
    GF = cell(ss,os,ps);
    minF = zeros(ss,os,ps);
    maxF = zeros(ss,os,ps);

    disp(['Filtering image: ',file]);
    disp(['Orientations: ',int2str(rad2deg(orient)), ' (degrees)']);
    disp(['Wavelengths: ',int2str(scale), ' (pixels)']);
    disp(['Phases: ',int2str(rad2deg(psi)), ' (degrees)']);

    I = imread(file);

    if strcmp(finfo.ColorType, 'truecolor') % Convert to grayscale
        I = rgb2gray(I);
    end

    if ~isa(I,'double') 
        I = double(I);
    end

    %figure; imshow(I); title('Original Image');

    % Create a new image of twice the dimensions
    % and copy the original image into the center.
    % This resulting image is then convolved, and
    % we only keep convolution result values for the
    % part of the image where the original image 
    % is located.

    % Make padded image, set to provided background color
    paddedImage = paddingColor * ones(paddedImageD(1),paddedImageD(2));

    % Copy original image into center of padded image
    top = paddedImageD(1)/2 - finfo.Height/2;
    left = paddedImageD(2)/2 - finfo.Width/2;
    paddedImage(left + (1:finfo.Height), top + (1:finfo.Width)) = I;

    %figure; imshow(paddedImage); title('Padded Image');

    % COMMENTED OUT
    % 03.06.11 by Bedeho Mender
    % Has no impact since sum of filter is 0
    %Remove DC component - subtract mean pixel intensity (see Pinto et al. 2008 & Deco)
    %This is weighted by the Gaussian envelope in Petkov & Kruizinga 1996
    %paddedImage = paddedImage - mean2(paddedImage); % mean(mean(I)); % No effect since the sum of a filter is 0

    % COMMENTED OUT
    % 03.06.11 by Bedeho Mender
    % Not meaningfull to take std since
    % mean is always zero, so is just quirky norm (1/(N-1) factor),
    % We do it once later, with proper norm.
    %disp(['Std dev of the image: ',num2str(std2(I))]);
    %if (std2(paddedImage))
    %    paddedImage = paddedImage / std2(paddedImage); % Divide by standard deviation to give unit variance
    %end

    %figure; imshow(paddedImage); title('Normalized Image');

    if set
        [path,name,ext] = fileparts(file);
        clear ext ver;
        outputDir = ['' path, filesep, name '.flt'];%sprintf('%s.flt',name); 

        % If we got an old folder, then just remove it!
        % We may be running with new params or something.
        if exist(outputDir, 'dir'),
            rmdir(outputDir, 's');
        end

        mkdir(outputDir);
    end
    
    for p=1:ps

        if ~set
            fi=figure(); suptitle('Gabor Filters');
            ci=figure(); suptitle('Filtered Images');
        end

        for s=1:ss
            for o=1:os

                % Create Gabor Filter
                GF{s,o,p} = double(gabor_fn(bw,gamma,psi(p),scale(s),orient(o)));

                % COMMENTED OUT
                % 03.06.11 by Bedeho Mender
                % In theory this is pointless, since filter sum is 0,
                % however the discrete and finite part GF{s,o,p}
                % does not, so we just give it zero sum by subtracting
                % mean.
                %Set Gabor filter mean to 0
                GF{s,o,p} = GF{s,o,p} - mean2(GF{s,o,p}); 

                % COMMENTED OUT
                % 03.06.11 by Bedeho Mender
                % We do normalization after convolution
                %Normalize gabor filter
                GF{s,o,p} = GF{s,o,p} / norm(GF{s,o,p}(:));

                if ~set
                    figure(fi);
                    subplot(ss,os,((s-1)*os)+o);
                    imshow(GF{s,o,p}/2+0.5);
                end
                
                % Convolve padded image with Gabor filter
                tmp = conv2(paddedImage, GF{s,o,p}, 'same');
                %%% DO NOT USE tmp = normalize_SIMON(paddedImage, tmp, length(GF{s,o,p}));
                
                %tmp = conv_SIMON(paddedImage, GF{s,o,p}); 
                
                % Copy out part of padded image convolution that corresponds to
                % the original image
                FI{s,o,p} = tmp((1:originalImageD(1)) + (paddedImageD(1)-originalImageD(1))/2 ,(1:originalImageD(2)) + (paddedImageD(2)-originalImageD(2))/2);

                % COMMENTED OUT
                % 03.06.11 by Bedeho Mender
                % Our normalization takes care of it
                % Normalize with selfconvolution of gabor filter to control for
                % varying size of gabor filer for diffrent parameter
                % combinations
                %FI{s,o,p}=FI{s,o,p}/max(max(abs(conv2(GF{s,o,p},GF{s,o,p}))));

                % COMMENTED OUT
                % 03.06.11 by Bedeho Mender
                % Our normalization takes care of it 
                %Give unit variance
                %FI{s,o,p} = FI{s,o,p} / std2(FI{s,o,p});
                
                FI{s,o,p} = FI{s,o,p} / norm(FI{s,o,p});
                
                % Cut away negatives
                FI{s,o,p}(FI{s,o,p} < 0) = 0;
                
                if set % Save file
                    filtFileName = ['' outputDir filesep name '.' int2str(scale(s)) '.' int2str(rad2deg(orient(o))) '.' int2str(rad2deg(psi(p))) '.gbo' '']; %sprintf('%s',outputDir);
                    outFile = fopen(filtFileName,'w');
                    count = fwrite(outFile, FI{s,o,p}','float'); % Transpose to make row major
                    %disp(FI{s,o,p});

                    if count ~= fsize
                        error('Filter size mismatch');
                    end

                    fclose(outFile);
                end

                % Display minimum and maximum convolved image values
                minF(s,o,p) = min(min(FI{s,o,p}));
                maxF(s,o,p) = max(max(FI{s,o,p}));

                if ~set % Plot filtered image
                    figure(ci);
                    subplot(ss,os,((s-1)*os)+o);
                    imshow(FI{s,o,p});
                    %imshow(FI{s,o,p}/2+0.5);
                end
            end
        end
    end

    %disp(minF);
    %disp(maxF);
end

% Simons new convolution routine
% not vectorized
% filter and image must be square
% filter must have odd dimension
% TOOO SLOW
function res = conv_SIMON(image, filter)

    imageDim = length(image);
    filterDim = length(filter);
    hFltDim = floor(filterDim/2);
    res = zeros(imageDim);
    
    fVector = filter(:);
    imgSpace = (hFltDim + 1):(imageDim-hFltDim);
    fNorm = norm(fVector);
    fltBase = -hFltDim:hFltDim;
    
    % Pick image position to apply filter on
    for imgRow=imgSpace,
        for imgCol=imgSpace,
            
            img = image(imgRow + fltBase, imgCol + fltBase);
            patch = img(:);
            imgNorm = norm(patch);
            
            res(imgRow, imgCol) = dot(patch, fVector)/(fNorm * imgNorm);
        end
    end
end

% NOT TESTED
%{
function res = normalize_SIMON(image, convResult, filterDim)

    imageDim = length(image);
    hFltDim = floor(filterDim/2);
    res = zeros(length(convResult));
    
    imgSpace = (hFltDim + 1):(imageDim-hFltDim);
    fltBase = -hFltDim:hFltDim;
    
    % Pick image position to apply filter on
    for imgRow=imgSpace,
        for imgCol=imgSpace,
            
            img = image(imgRow + fltBase, imgCol + fltBase);
            
            res(imgRow, imgCol) = convResult(imgRow, imgCol)/norm(img(:));
        end
    end
end
%} 
