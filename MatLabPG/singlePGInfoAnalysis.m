
experimentName = '1.4--FF_FB_LAT_stdp_0.005'
timestep = 0.00002;
presentationTime = 2.0;
nObjs = 3;
nTrans = 2;
T = 50;
jitterSize = 0.005;
plotEachPG=0;

ExcitDim = 64;
InhibDim = 32;
nLayers = 4;
N = (ExcitDim*ExcitDim+InhibDim*InhibDim)*nLayers;% N: num neurons

anchorWidth = 3;


for trainedNet = [1]
    if(trainedNet)
        load(['../output/' experimentName '/groups_trained.mat']);
        fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Trained_Epoch0.bin']);
        fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Trained_Epoch0.bin']);
    else
        load(['../output/' experimentName '/groups_untrained.mat']);
        fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Untrained_Epoch0.bin']);
        fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Untrained_Epoch0.bin']);
    end
    
    spikes_id = fread(fileID_id,'int32');
    fclose(fileID_id);
    
    spikes_time = fread(fileID_time,'float32');%second
    fclose(fileID_time);
    
    triggerCountMatrix = zeros(nObjs,nTrans,length(groups));
    activatedPGIDs = [];
    activatedPGTimes = [];
    
    for trigger_index = 1:length(groups)
%        firings = groups{1,trigger_id}.firings;
        index_t = 1;
        trigger_ids = groups{1,trigger_index}.gr(1:anchorWidth,2);
        trigger_times = (groups{1,trigger_index}.gr(1:anchorWidth,1)-1)*timestep;

%         trigger_ids = groups{1,trigger_index}.firings(1:anchorWidth,2);
%         trigger_times = (groups{1,trigger_index}.firings(1:anchorWidth,1)-1)*timestep;
%         disp(round(trigger_index*100.0/length(groups)));
        for obj=0:nObjs-1
            for trans=0:nTrans-1
%                 disp([trigger_index, obj, trans]);
                time_benig = (obj*nTrans+trans)*presentationTime;
                time_end = time_benig + presentationTime;
                cond = (spikes_time>=time_benig) & (spikes_time<time_end);
                id_subset = spikes_id(cond);
                time_subset = spikes_time(cond);
                
                for i=1:length(id_subset)
                    if id_subset(i)==trigger_ids(1)
                        flag = true;
                        for a=2:anchorWidth
                            timimg = time_subset(i)+trigger_times(a);
                            cond2 = (id_subset==trigger_ids(a)) & (abs(time_subset-timimg)<jitterSize);
                            if(length(id_subset(cond2))==0)
                                flag = false;
                                break;
                            end
                        end
                        if(flag)
                            triggerCountMatrix(obj+1,trans+1,trigger_index) = triggerCountMatrix(obj+1,trans+1,trigger_index)+1;
                            disp(['Found! -- trigger:' num2str(trigger_index) ' obj:' num2str(obj) ' trans:' num2str(trans)]);
                            activatedPGIDs = [activatedPGIDs trigger_index];
                            activatedPGTimes = [activatedPGTimes time_subset(i)]; 
                        end
                    end
                end

                
                id_subset;
                
                
                
%                 while(spikes_time(index_t)<time_end)
%                     if(spikes_id(index_t)==trigger_ids(1))
%                         t_init = spikes_time(index_t);
%                          
%                     end
%                     index_t=index_t+1;
%                 end

                
            end
        end
    end
    
    nBins = 2;
    IRs = infoAnalysis_PG(triggerCountMatrix,trainedNet,nBins);
    
    

    

    for obj=1:nObjs
        figure(1)
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
        
        if obj==1
            mark = 'ro';
        elseif obj==2
            mark = 'g^';
        else
            mark = 'bs';
        end
        plot(time_subset,id_subset,mark);
%         plot(time_subset,id_subset,mark,'color','k');
        hold on;
        
        for t =1:nTrans-1
            plot([presentationTime*((obj-1)*nTrans + t) presentationTime*((obj-1)*nTrans + t)],[-500 length(triggerCountMatrix)+500],'--k','LineWidth',2);
        end
        if 1<obj && obj<=nObjs
            plot([presentationTime*((obj-1)*nTrans) presentationTime*((obj-1)*nTrans)],[-500 length(triggerCountMatrix)+500],'-k','LineWidth',2);
        end
        
        if(plotEachPG==1)
            fig=figure(2);        
            for i_PG = 1:length(indexPGwithHighInfo)
                gr = groups{1,indexPGwithHighInfo(i_PG)}; 

                plot(gr.firings(:,1),gr.firings(:,2),'o');
                strValues = strtrim(cellstr(num2str([gr.firings(:,2), gr.firings(:,1)],'(%d, %3.1f)')));
                text(gr.firings(:,1),gr.firings(:,2),strValues,'VerticalAlignment','bottom');
                hold on;

                for l=1:nLayers
                    plot([0 T],[(ExcitDim*ExcitDim+InhibDim*InhibDim)*l (ExcitDim*ExcitDim+InhibDim*InhibDim)*l],'k');
                    plot([0 T],[(ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1) (ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1)],'k--');
                end
                for j=1:size(gr.gr,1)
                    if gr.gr(j,6)==1
                        lineCol = [0.5 0.5 0.5];
                    else
                        lineCol = [0,0,0];
                    end
                    plot(gr.gr(j,[1 3 5]),gr.gr(j,[2 4 4]),'.-','LineWidth',gr.gr(j,6),'color',lineCol);
                end;
                axis([0 T 0 N]);
                hold off
                title(['group id:' num2str(indexPGwithHighInfo(i_PG))]);
                ylabel('Cell Index');
                xlabel('Time [ms]')
                drawnow;


                saveas(fig,['../output/' experimentName '/polyPlot/obj_' num2str(obj) '_poly_i_' num2str(indexPGwithHighInfo(i_PG)) '.fig']);
                set(gcf,'PaperPositionMode','auto')
                print(['../output/' experimentName '/polyPlot/_obj_' num2str(obj) '_poly_i_' num2str(indexPGwithHighInfo(i_PG))],'-dpng','-r0');
            end
        end 
    end
    
    figure(1);
    ylim([-500 length(triggerCountMatrix)+500]);
    title('Stimulus Specific PGs');
    ylabel('Index of Polychronous Groups');
    xlabel('Time [s]');
end


