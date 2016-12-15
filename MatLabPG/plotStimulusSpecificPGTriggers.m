
cell_id_time = cell(3,nObjs);
cumPGs = 0;
for obj=1:nObjs
    [B,I] = sort(IRs(:,obj),'descend');

    indexPGwithHighInfo = I(B>log2(nObjs)*0.9);
    cond = ismember(activatedPGIDs,indexPGwithHighInfo);
    id_subset = activatedPGIDs(cond);
    time_subset = activatedPGTimes(cond);

    %remove cells that represent stimulus by not responding
    timeBegin = presentationTime*nTrans*(obj-1);
    timeEnd = presentationTime*nTrans*(obj);
    PGindexOutRange = unique(id_subset(time_subset<timeBegin | timeEnd<time_subset));
    cond = ~ismember(id_subset,PGindexOutRange);
    id_subset = id_subset(cond);
    time_subset = time_subset(cond);
    
    
    
    cell_id_time{1,obj} = id_subset;
    cell_id_time{2,obj}=time_subset;
    
    id_subset_reoreder=id_subset;
    
    ids_uniq= unique(cell_id_time{1,obj});
    for i=1:length(id_subset)
        id_subset_reoreder(i) = find(ids_uniq==id_subset(i))+cumPGs;
    end

    cell_id_time{3,obj} = id_subset_reoreder;
    
    cumPGs=cumPGs+length(ids_uniq)
    
%     plot(cell_id_time{2,obj},cell_id_time{3,obj},'.')
%     hold on;
    
end

rasterTable = zeros(cumPGs,presentationTime*nTrans*nObjs*100);
figure;
hold on;
for obj=1:nObjs
    for i=1:length(cell_id_time{3,obj})
%         rasterTable(cell_id_time{3,obj}(i),int32(cell_id_time{2,obj}(i)*100))=rasterTable(cell_id_time{3,obj}(i),int32(cell_id_time{2,obj}(i)*100))+1;
        rasterTable(cell_id_time{3,obj}(i),int32(cell_id_time{2,obj}(i)*100))=1;

    end
end

rasterMax = max(rasterTable(:));
rasterTable = rasterTable./rasterMax;

imagesc(rasterTable);

c_w2r = 1-gray;
c_w2r(:,1)=1.0;
% c_w2r(:,2:3)=1-c_w2r(:,2:3);
colormap(c_w2r)

for obj=1:nObjs
    for t =1:nTrans-1
        plot([presentationTime*((obj-1)*nTrans + t)*100 presentationTime*((obj-1)*nTrans + t)*100],[1 cumPGs],'--k','LineWidth',2);
    end
    if 1<obj && obj<=nObjs
        plot([presentationTime*((obj-1)*nTrans)*100 presentationTime*((obj-1)*nTrans)*100],[1 cumPGs],'-k','LineWidth',2);
    end
end
ylim([1 cumPGs]);
xlim([1 presentationTime*nObjs*nTrans*100]);
title('Stimulus Specific PGs');
ylabel('Index of Polychronous Groups');
xlabel('Time [s]');

