function IRs = infoAnalysis(inputMatrix,trained,nBins)
    plotAllSingleCellInfo = 0;
%     hold off;
%    num_bins =  2;%num_transforms;   %can be adjusted
    weightedAnalysis = 1;%exclude the selectivity by not responding to a particular stimulus


    num_transforms = size(inputMatrix,2);
    num_stimulus = size(inputMatrix,1);
    num_PGs = size(inputMatrix,3);

    showAllIRs = zeros(num_stimulus,num_PGs);



    disp('** Data loading **');

    sumPerBin = zeros(num_PGs,nBins);
    sumPerObj = num_transforms;
    sumPerCell = num_transforms*num_stimulus;
    IRs = zeros(num_PGs,num_stimulus);   %I(R,s) single cell information
    IRs_weighted = zeros(num_PGs,num_stimulus);   %I(R,s) single cell information

    binMatrix = zeros(num_PGs, num_stimulus, nBins); %number of times when fr is classified into a specific bin within a specific objects's transformations

    
    if max(inputMatrix(:))>0.001
        FR_tmp = inputMatrix./max(inputMatrix(:));
    end

    
    for obj = 1:num_stimulus;
        disp([num2str(obj) '/' num2str(num_stimulus)]);
        for trans = 1:num_transforms;
            for i = 1:num_PGs;
                bin = min(floor(FR_tmp(obj,trans,i))*nBins,nBins-1)+1;
                binMatrix(i,obj,bin)=binMatrix(i,obj,1)+1;
            end
        end
    end


    disp('DONE');
    disp(['** single-cell information analysis **']);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % single-cell information analysis      %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Loop through all cells to calculate single cell information
    for i=1:num_PGs
        % For each cell, count the number of transforms per bin
        for bin=1:nBins
            sumPerBin(i,bin)=sum(binMatrix(i,:,bin));
        end

        % Calculate the information for cell_x cell_y per stimulus
        for obj=1:num_stimulus
            for bin=1:nBins
                Pr = sumPerBin(i,bin)/sumPerCell;
                Prs = binMatrix(i,obj,bin)/sumPerObj;
                if(Pr~=0&&Prs~=0&&Pr<Prs)
                    IRs(i,obj)=IRs(i,obj)+(Prs*(log2(Prs/Pr)));%*((bin-1)/(num_bins-1)); %could be added to weight the degree of firing rates.
                    %IRs(row,col,object)=IRs(row,col,object)+(Prs*(log2(Prs/Pr)))*((bin-1)/(num_bins-1)); %could be added to weight the degree of firing rates.
                    IRs_weighted(i,obj)=IRs_weighted(i,obj)+(Prs*(log2(Prs/Pr)))*((bin-1)/(nBins-1)); %could be added to weight the degree of firing rates.
                end
            end
        end
    end

    if (weightedAnalysis==1)
        IRs = IRs_weighted;
    end

    % Order by information content, descending
    IRs_tmp = sort(max(IRs,[],2), 'descend');%find max IRs for each 
    

    for obj = 1:num_stimulus
            tmp = [1:num_PGs;reshape(IRs(:,obj),1,[])];
            sorted = transpose(sortrows(transpose(tmp),-2));
    end
    %tmpstore = performanceArray;

    figure(1)
    hold on;
    for no = 1:num_stimulus
        showAllIRs(no,:) = sort(IRs(:,no), 'descend');
    end
    
    
    if(trained)
        if(plotAllSingleCellInfo)
            plot(transpose(showAllIRs), 'k-','LineWidth',2);
        else
            plot(IRs_tmp,'k-','LineWidth',2);
        end
    else
        if(plotAllSingleCellInfo)
            plot(transpose(showAllIRs), 'k--','LineWidth',2);
        else
            plot(IRs_tmp,'k--','LineWidth',2);
        end
    end
    
    
    %axis([0 num_PGs/10 -0.1 log2(num_stimulus)+0.1]);
    nCellsWithNonZeroInfo = length(find(IRs>0));
    axis([1 nCellsWithNonZeroInfo*1.2 -0.1 log2(num_stimulus)+0.1]);
    title('Single Cell Information Analysis');
    ylabel('Information [bit]');
    xlabel('Cell Rank');
    hold on;


%     dlmwrite([experiment '/IRs' num2str(layer) '.csv'],IRsHist);
%     dlmwrite([experiment '/IRs' num2str(layer) 'All_s' num2str(session) '.csv'],showAllIRs);
%     if(plotAllSingleCellInfo)
%         saveas(fig,[experiment '/info_single_all' num2str(layer) '.png']);
%         saveas(fig,[experiment '/info_single_all' num2str(layer) '.fig']);
%     else
%         saveas(fig,[experiment '/info_single' num2str(layer) '.png']);
%         saveas(fig,[experiment '/info_single' num2str(layer) '.fig']);
%     end


end


