function [] = filterImageSet(inputDirectory, nrOfObjects, nrOfTransforms, imageSize, filterScale, paddingGrayScaleColor)

%inputDirectory = Directory where we look for /Images folder with
%input pictures, and where we save /Filtered folder with output and
%files FileList.txt and FilteredParameters.txt
% NO TRAILINGNSLASH
%
%imageSize = is the size you expect images to be, this is tested
%against real image size prior to filtering as safety, been burned
%to many times!!!

tic;
imageParamFile = 'FilterParameters.txt';
imageListFile = 'FileList.txt';
%paddingGrayScaleColor = 128; % Perfect gray is 127.5, so 127 and 128 work
%equally well, but make sure that it matches background of stimuli
outdir = [inputDirectory '/Filtered'];
set = true;
ext='.png';
token = '*';

% READ: http://www.cs.rug.nl/~imaging/simplecell.html 
psi   = [0, pi, -pi/2, pi/2];       % phase, [0, pi, -pi/2, pi/2]
scale = filterScale;                % wavelength (pixels)
orient= [0, pi/4, pi/2, 3*pi/4];    % orientation
bw    = 1.5;                        % bandwidth
gamma = 0.5;                        % aspect ratio

% Create parameters file ===========================================

vPhases = ['[' num2str(rad2deg(psi(1)))];
for p = 2:length(psi)
    vPhases = [vPhases ', ' num2str(rad2deg(psi(p)))];
end
vPhases = [vPhases ']'];

vScales = ['[' num2str(scale(1))];
for s = 2:length(scale)
    vScales = [vScales ', ' num2str(scale(s))];
end
vScales = [vScales ']'];

vOrients = ['[' num2str(orient(1))];
for o = 2:length(orient)
    vOrients = [vOrients ', ' num2str(rad2deg(orient(o)))];
end
vOrients = [vOrients ']'];

iParams = fopen([inputDirectory,'/',imageParamFile],'w+');
fprintf(iParams, ['vPhases = ' vPhases ' ;\n'] );
fprintf(iParams, ['vScales = ' vScales ' ;\n'] );
fprintf(iParams, ['vOrients = ' vOrients ' ;\n'] );

code = fclose(iParams);

if code ~= 0
    error('Problem closing iParams');
end

% Save results for summary ===========================================

fileID = fopen([inputDirectory filesep 'Summary.html'], 'w'); 
fprintf(fileID, '<h1>Filtering - %s</h1>\n', datestr(now));

fprintf(fileID,  '<ul>\n');
fprintf(fileID, ['<ul> Phases - ' vPhases '</ul>\n']);
fprintf(fileID, ['<ul> Scales - ' vScales '</ul>\n']);
fprintf(fileID, ['<ul> Orients - ' vOrients '</ul>\n']);
fprintf(fileID,  '</ul>\n');

fprintf(fileID, '<table cellpadding="10" style="border: solid 1px">\n');
fprintf(fileID, '<tr> <th>Image</th> <th>Filter</th> <th>Image</th> </tr>\n');

% Filter & create file list and summary ========================================

iList = fopen([inputDirectory,'/',imageListFile],'w+');

content = dir([inputDirectory,'/Images']);
cell = struct2cell(content);
[~,ind] = sort_nat(cell(1,:)); % Sort numerically according to first row (name)
clear cell;% S;

filesOutputted = 0;

for i = 1:length(content)
    file = content(ind(i)).name;
    
    if ~(content(ind(i)).isdir)
        
        ignoreDir = any(strcmpi(file, {'private','CVS','.','..'}));
        ignorePrefix = any(strncmp(file, {'@','.'}, 1));
        
        [pathstr, fname, fext] = fileparts(file);
        
        if (~(ignoreDir || ignorePrefix) && strcmpi(ext,fext))
            
            % Open file
            imgFile = [inputDirectory filesep 'Images' filesep file];
            finfo = imfinfo(imgFile);
            
            % Test for expected image size
            if finfo.Height ~= imageSize || finfo.Width ~= imageSize,
                error(['UNEXPECTED IMAGE SIZE FOUND: ' finfo.Height ',' finfo.Width]);
            end
            
            % Filter
            filtering(imgFile, psi, scale, orient, bw, gamma, set, paddingGrayScaleColor);
            
            % Dump to file list
            fprintf(iList, '%s\n', fname);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Workaround James wanted, it is required for infoanalysis to
            % work: if each object as one transform, then add duplicate
            if nrOfTransforms == 1,
                fprintf(iList, '%s\n', fname);
            end            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Output star if we are going from one object to the next
            filesOutputted = filesOutputted + 1;
            if mod(filesOutputted, nrOfTransforms) == 0,
                fprintf(iList, [token '\n']);
            end
            
            % Dump to summary            
            viewFilters = ['matlab:plotFilters(\''' inputDirectory filesep 'Filtered' filesep fname '\'',' num2str(imageSize) ',' vOrients ',' vPhases ',' vScales ')'];
            viewImage = ['matlab:figure;imshow(\''' imgFile '\'')'];
            
            fprintf(fileID, '<tr>\n');
            fprintf(fileID, '<td>%s</td>\n', fname);
            fprintf(fileID, '<td><input type="button" value="View" onclick="document.location=''%s''"/></td>\n', viewFilters);
            fprintf(fileID, '<td><input type="button" value="View" onclick="document.location=''%s''"/></td>\n', viewImage);
            fprintf(fileID, '</tr>\n');
        end
    end
end

% Check
if nrOfTransforms * nrOfObjects ~= filesOutputted,
  error('The number of files in file list does not equal nrOfTransforms * nrOfObjects\n');   
end

code = fclose(iList);
if code ~= 0; error('Problem closing file list'); end

fprintf(fileID, '</table>');
fclose(fileID);

% Cleanup ========================================

toc;

% Delete existing output folder, make new
if exist(outdir, 'dir'),
    rmdir(outdir, 's');
end

mkdir(outdir);
    
cmd = ['mv ' inputDirectory '/Images/*.flt ' outdir];
[status, results] = system(cmd); % 2nd return value is the output of the cmd (results)
if (status ~= 0)
    disp(cmd);
    disp(results); 
    error('Problem moving filtered folders to /Filtered directory'); 
end

web([inputDirectory filesep 'Summary.html']);
