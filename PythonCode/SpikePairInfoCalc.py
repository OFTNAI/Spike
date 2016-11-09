import numpy as np
import pylab as plt
import pickle
import os;
import math;
from scipy import signal


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    
    if(math.sqrt((a*a).sum() * (b*b).sum())!=0):
        r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    else:
        r = (a*b).sum();
    return r        

# experimentName = '20160904_FF_successful';
# experimentName = '20160908_FF_LAT';
experimentName = '1.5--FF_FB_LAT_stdp_0.005_nCon5_seed1';
phases = ['Untrained','Trained'];
# phases = ['Untrained'];
# phases = ['Trained'];

nObj = 3;
nTrans = 2;
nLayers = 4;
presentationTime = 2000;
exDim = 64;
inDim = 32;
nExcitCells = exDim*exDim;
nInhibCells = inDim * inDim;
maxDelay = 10;

mimNumSpikeTh = 10;
# minSizeSpikeChain = 2;
SpikeChainDetectionProbTh = 0.2;
nPairsPlotForInfo = 10000;

nBins = 3;

# jitter = 1;#ms

fig=plt.figure(4 , figsize=(20, 5),dpi=150);

# I = np.zeros((nObj*nTrans,nObj*nTrans));
# for obj in range(nObj):
#     for index_r in range(obj*nTrans,obj*nTrans+nTrans):
#         for index_c in range(obj*nTrans,obj*nTrans+nTrans):
#             I[index_r,index_c] = 1.0;


phaseIndex=1;
for phase in phases:
    fn_id = "../output/"+experimentName+"/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
    fn_t = "../output/"+experimentName+"/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  
    dtIDs = np.dtype('int32');
    dtTimes = np.dtype('f4');
     
    spikeIDs = np.fromfile(fn_id, dtype=dtIDs);
    spikeTimes = (np.fromfile(fn_t, dtype=dtTimes)*1000).astype(int);
    
    cond = spikeTimes>0;
    spikeIDs = np.extract(cond,spikeIDs);
    spikeTimes = np.extract(cond, spikeTimes)

    for l in range(0,nLayers):
#         l = 3;
        print(phase + " -- layer: "+str(l));
        #     maxSCindex=np.zeros((nObj,nTrans,4,nExcitCells));
        #     probTable_list = np.zeros((nObj,nTrans,nExcitCells,nExcitCells,maxDelay));#targetcell, 
        spikePair_maxNum = 1000000;
        spikePairDetector = np.zeros((nObj,nTrans,spikePair_maxNum));#targetcell, 
        
#         l = 3;
        
        
        #extract spike data of a speficied layer
        cond_ids = (l*(nExcitCells+nInhibCells) < spikeIDs) & (spikeIDs < l*(nInhibCells+nExcitCells)+nExcitCells) & (spikeTimes<presentationTime*nObj*nTrans);
        spikeIDs_layer = np.extract(cond_ids, spikeIDs)-l*(nExcitCells+nInhibCells);
        spikeTimes_layer = np.extract(cond_ids, spikeTimes);
         
        
        #create a list of id of cells that spike at a particular timing
        ListOfCellsThatSpikeAtT = [];
        for t in range(presentationTime*nObj*nTrans):
            ListOfCellsThatSpikeAtT.append([]);
        for index in range(spikeIDs_layer.size):
            ListOfCellsThatSpikeAtT[int(spikeTimes_layer[index])].append(spikeIDs_layer[index]);
        
        
        SpikeChain_string = [];
#         SpikeChain = []; #(obj,trans,spikechains_index)
        SpikePair = [];
        SpikePair_string = [];
        count = 0;
        for obj in range(nObj):
#             SpikeChain.append([]);
            SpikePair.append([]);
            for trans in range(nTrans):
                probTable_list = np.zeros((nExcitCells,nExcitCells,maxDelay));#targetcell, 
                
#                 SpikeChain[obj].append([]);
                SpikePair[obj].append([]);
    
                print('time: [' + str((presentationTime*(obj*nTrans+trans))) + ', ' + str(presentationTime*(obj*nTrans+trans+1)) + ') -- obj:' +str(obj) + ', trans:' + str(trans));
                
                #extract spikes during the presentation of a specific object obj at specific transform trans.
                cond_perStim = ((presentationTime*(obj*nTrans+trans)) <= spikeTimes_layer) & (spikeTimes_layer < (presentationTime*(obj*nTrans+trans+1)));
                spikeIDs_perStim = np.extract(cond_perStim,spikeIDs_layer);
                spikeTimes_perStim = np.extract(cond_perStim,spikeTimes_layer);
                
                spikeCountPerCell = np.zeros(exDim*exDim);
                for index_spike in range(spikeIDs_perStim.size):#for each target stimulus
                    targetSpikeID = spikeIDs_perStim[index_spike];
                    spikeCountPerCell[targetSpikeID]+=1;
    #                     t_begin = max(presentationTime*(obj*nTrans+trans),spikeTimes_perStim[index_spike]-maxDelay);
                    t_begin = spikeTimes_perStim[index_spike]-maxDelay;
                    if spikeTimes_perStim[index_spike]>presentationTime*(obj*nTrans+trans)+maxDelay:
                        for t in range(t_begin,spikeTimes_perStim[index_spike]):
                            listOfCells_tmp = np.extract(ListOfCellsThatSpikeAtT[t]!=targetSpikeID,ListOfCellsThatSpikeAtT[t]);#to ignore self excitement
                            if listOfCells_tmp.size>0:
                                probTable_list[targetSpikeID,listOfCells_tmp,t-t_begin]+=1.0;
                 
                 
                for targetSpikeID in range(exDim*exDim):
                    if (spikeCountPerCell[targetSpikeID]<mimNumSpikeTh):#if the number of spikes of target cell was not satisfactory 
                        probTable_list[targetSpikeID] = 0;
                    else:
                        probTable_list[targetSpikeID]/=spikeCountPerCell[targetSpikeID];
                        
                    tmp = np.where(probTable_list[targetSpikeID]>(1.0/nBins));
    #                     if (tmp[0].size>=minSizeSpikeChain):
    #                         sortedIndex = np.argsort(tmp[1])
    #                         reordered = np.array(tmp)[:,sortedIndex];
    #                         SpikeChain[obj][trans].append([]);
    #                         spikeTimeOfTargetCell = maxDelay-reordered[1].min();#calculate spike timings
    #                         for spike_index in range(reordered[1].size):
    #                             SpikeChain[obj][trans][len(SpikeChain[obj][trans])-1].append([reordered[0,spike_index],reordered[1,spike_index]-reordered[1].min()]);
    #                         SpikeChain[obj][trans][len(SpikeChain[obj][trans])-1].append([targetSpikeID,spikeTimeOfTargetCell]);
    #                         
    # #                         print('targetID:' + str(targetSpikeID) + str(SpikeChain[obj][trans][len(SpikeChain[obj][trans])-1]));
    #                         spikeChainStr = str(SpikeChain[obj][trans][len(SpikeChain[obj][trans])-1]);
    #                         if spikeChainStr not in SpikeChain_string:
    #                             SpikeChain_string.append(str(SpikeChain[obj][trans][len(SpikeChain[obj][trans])-1]));
                     
                    for tmp_index in range(tmp[0].size):
                        SpikePair[obj][trans].append([tmp[0][tmp_index],maxDelay - tmp[1][tmp_index],targetSpikeID]);
                        spikePairStr = str(SpikePair[obj][trans][len(SpikePair[obj][trans])-1]);
    #                         print(spikePairStr);
                        if spikePairStr not in SpikePair_string:
                            SpikePair_string.append(spikePairStr);
    #                         spikePairDetector[obj,trans,l,len(SpikePair_string)-1]=1;
                            spikePairDetector[obj,trans,len(SpikePair_string)-1]=probTable_list[targetSpikeID,tmp[0][tmp_index],tmp[1][tmp_index]];
                        else:
                            spikePairDetector[obj,trans,SpikePair_string.index(spikePairStr)]=probTable_list[targetSpikeID,tmp[0][tmp_index],tmp[1][tmp_index]];
                        
                            
    #                         ListOfSignificantSC.append(targetSpikeID);
                count+=len(SpikePair[obj][trans]);
    #                 print(str(SpikePair[obj][trans]));
    #                 print('new SpikeChain:' + str(len(SpikeChain[obj][trans])) + ' total:' + str(count) +' uniequ total:' + str(len(SpikeChain_string)));
                print('new SpikePairs:' + str(len(SpikePair[obj][trans])) + ' total:' + str(count) +' unique total:' + str(len(SpikePair_string)) + '\n');
    #                 for plot_i in range(1,21):
    #                     plt.subplot(2,10,plot_i)
    #                     plt.imshow(probTable_list[obj,trans,ListOfSignificantSC[plot_i]], extent=[0,maxDelay,0,nExcitCells], aspect='auto', interpolation='none');
    #                     plt.colorbar();
    #         
    #                 colorBarOn = False;
    
    
        
    
        plotAllSingleCellInfo = False;
        weightedAnalysis = False;
        showImage = False;
        saveImage = True;
        
#         nPairs = nPairsPlotForInfo;#len(SpikePair_string);
        nPairs = len(SpikePair_string);
        #infor analysis
        infos = np.zeros(nPairs);
                    
        sumPerBin = np.zeros((nPairs,nBins));
        sumPerObj = nTrans;
        sumPerCell = nTrans*nObj;
        IRs = np.zeros((nObj,max(nPairs,nPairsPlotForInfo)));#I(R,s) single cell information
        IRs_weighted = np.zeros((nObj,max(nPairs,nPairsPlotForInfo)));#I(R,s) single cell information
        pq_r = np.zeros(nObj)#prob(s') temporally used in the decoing process
        Ps = 1/nObj   #Prob(s) 
        
        
        print("**Loading data**")
        binMatrix = np.zeros((nPairs, nObj, nBins));# #number of times when fr is classified into a specific bin within a specific objs's transformations
        for obj in range(nObj):
            print str(obj) + '/' + str(nObj);
            for trans in range(nTrans):
                for cell in range(nPairs):
    #                             bin = np.around(FR_tmp[obj,trans,l,cell]*(nBins-1));
                    bin = min(np.floor((spikePairDetector[obj,trans,cell])*(nBins)),nBins-1)
                    binMatrix[cell,obj,bin]=binMatrix[cell,obj,bin]+1;
    #                             
    #                             
        
        
        #print binMatrix;
    #                 for obj in range(nObj):
    #                     print str(obj) + '/' + str(nObj);
    #                     for trans in range(nTrans):
    #                         for cell in range(nExcitCells):                            
    # 
    #                             hist = np.histogram(FR[obj,:,l,cell], bins=range(nBins+1), range=(0,maxFRTh));
    #                             for bin in range(nBins):
    #                                 binMatrix[cell,obj,bin]=hist[0][bin];
                            
        
        
        print "** single-cell information analysis **";
        # Loop through all cells to calculate single cell information
        for cell in range(nPairs):
            # For each cell, count the number of transforms per bin
            for bin in range(nBins):
                for obj in range(nObj):
                    sumPerBin[cell,bin]+=binMatrix[cell,obj,bin];
    
            # Calculate the information for cell_x cell_y per stimulus
            for obj in range(nObj):
                for bin in range(nBins):
                    Pr = sumPerBin[cell,bin]/sumPerCell;
                    Prs = binMatrix[cell,obj,bin]/sumPerObj;
                    if(Pr!=0 and Prs!=0 and Pr<Prs):
                        IRs[obj,cell]+=(Prs*(np.log2(Prs/Pr)));#*((bin-1)/(nBins-1)); #could be added to weight the degree of firing rates.
                        #IRs(row,col,obj)=IRs(row,col,obj)+(Prs*(log2(Prs/Pr)))*((bin-1)/(nBins-1)); #could be added to weight the degree of firing rates.
                        IRs_weighted[obj,cell]+=(Prs*(np.log2(Prs/Pr)))*((bin)/(nBins-1)); #could be added to weight the degree of firing rates.
     
        
        
        
        if (weightedAnalysis):
            IRs = IRs_weighted;
        
        
        IRs_sorted = np.sort(IRs*-1)*-1;
        np.savetxt("../output/"+experimentName+"/SingleCellInfo_SpikePairs_l" + str(l) + phase + ".csv",IRs_sorted, delimiter=',');
        
        
        if (plotAllSingleCellInfo):
            IRs_sorted = np.sort(IRs*-1)*-1;
            plt.subplot(2,nLayers,(phaseIndex-1)*nLayers+(l+1));
    
    #                     plt.plot(np.transpose(IRs_sorted), color='k');
            plt.plot(np.transpose(IRs_sorted));
            
            plt.ylim([-0.05, np.log2(nObj)+0.05]);
            plt.xlim([0, nPairs])
            plt.title("Layer " + str(l));
            plt.ylabel("Information [bit]");
            plt.xlabel("Cell Rank");
            
        else:
            IRs = np.max(IRs,axis=0);
            
            IRs_sorted = np.transpose(np.sort(IRs));
            reversed_arr = IRs_sorted[::-1]
        
            infos = reversed_arr;
            np.savetxt("../output/"+experimentName+"/SingleCellInfo_SpikePairs_l" + str(l) + phase + "_max.csv",infos, delimiter=',');

        
            plt.subplot(1,nLayers,l+1)
            if phaseIndex==1:
                plt.plot(np.transpose(infos), linestyle='--', color='k');
            elif phaseIndex==2:
                plt.plot(np.transpose(infos), linestyle='-', color='k');
            plt.ylim([-0.05, np.log2(nObj)+0.05]);
            plt.xlim([0, nPairsPlotForInfo])
            plt.title("Layer " + str(l));
            plt.ylabel("Information [bit]");
            plt.xlabel("Cell Rank");
            plt.hold(True);
#             plt.show()
    
        
        
    phaseIndex+=1;
    
if showImage:
    plt.show();
if saveImage:
    if (plotAllSingleCellInfo):
        fig.savefig("../output/"+experimentName+"/SingleCellInfo_SpikePairs_ALL.png");
        fig.savefig("../output/"+experimentName+"/SingleCellInfo_SpikePairs_ALL.eps");

#                 fig.savefig("../output/SingleCellInfo_ALL.eps");
    else:
        fig.savefig("../output/"+experimentName+"/SingleCellInfo_SpikePairs_MAX.png");
        fig.savefig("../output/"+experimentName+"/SingleCellInfo_SpikePairs_MAX.eps");
#                   fig.savefig("../output/SingleCellInfo_MAX.eps");              

    print("figure SingleCellInfo.png is exported in Results");

plt.close();
    
    
    
    
    
#     RSM = np.zeros((nObj*nTrans,nObj*nTrans));
#     
#     probTable_list/=probTable_list.max();
# #     for row in range(nExcitCells*nObj*nTrans):#row = obj*trans+i
# #         print(str(round(row*100.0/(nExcitCells*nObj*nTrans))));
# #         row_cell_i = row%(nExcitCells);
# #         row_obj_i = math.floor(1.0*row/(nExcitCells*nTrans))%nObj;
# #         row_trans_i = math.floor(1.0*row/nExcitCells)%nTrans;
# # #         print('row_cell_i:'+str(row_cell_i)+', row_obj_i:'+str(row_obj_i)+', row_trans_i:'+str(row_trans_i));
# #         if (probTable_list[row_obj_i,row_trans_i,row_cell_i].max()==0):
# #             continue;
# #         for col in range(nExcitCells*nObj*nTrans):
# #             col_cell_i = col%(nExcitCells);
# #             col_obj_i = math.floor(1.0*col/(nExcitCells*nTrans))%nObj;
# #             col_trans_i = math.floor(1.0*col/nExcitCells)%nTrans;
# # #             print('\trow_cell_i:'+str(col_cell_i)+', row_obj_i:'+str(col_obj_i)+', row_trans_i:'+str(col_trans_i));
# #             if(probTable_list[col_obj_i,col_trans_i,col_cell_i].max()==0):
# #                 continue;
# # #             print(corr2(np.reshape(probTable_list[row_obj_i,row_trans_i,row_cell_i], (nExcitCells,maxDelay+1)),np.reshape(probTable_list[col_obj_i,col_trans_i,col_cell_i], (nExcitCells,maxDelay+1))));
# #             RSM[int(row_obj_i*nTrans+row_trans_i),int(col_obj_i*nTrans+col_trans_i)]+=corr2(probTable_list[row_obj_i,row_trans_i,row_cell_i],probTable_list[col_obj_i,col_trans_i,col_cell_i]);
#     
#     labels=[];
#     for obj in range(nObj):
#         for trans in range(nTrans):
#             labels.append('obj:'+str(obj)+', trans:'+str(trans));
# 
# 
#     for row in range(nObj*nTrans):#row = obj*trans+i
#         print(str(round(row*100.0/(nObj*nTrans))));
#         row_trans_i = row%nTrans;
#         row_obj_i = math.floor(1.0*row/nTrans)%nObj;
# #         print('row_cell_i:'+str(row_cell_i)+', row_obj_i:'+str(row_obj_i)+', row_trans_i:'+str(row_trans_i));
#         if (probTable_list[row_obj_i,row_trans_i].max()==0):
#             continue;
#         for col in range(nObj*nTrans):
#             col_trans_i = col%nTrans;
#             col_obj_i = math.floor(1.0*col/nTrans)%nObj;
# #             print('\trow_cell_i:'+str(col_cell_i)+', row_obj_i:'+str(col_obj_i)+', row_trans_i:'+str(col_trans_i));
#             if(probTable_list[col_obj_i,col_trans_i].max()==0):
#                 continue;
# #             print(corr2(np.reshape(probTable_list[row_obj_i,row_trans_i,row_cell_i], (nExcitCells,maxDelay+1)),np.reshape(probTable_list[col_obj_i,col_trans_i,col_cell_i], (nExcitCells,maxDelay+1))));
#             RSM[int(row_obj_i*nTrans+row_trans_i),int(col_obj_i*nTrans+col_trans_i)]+=corr2(probTable_list[row_obj_i,row_trans_i],probTable_list[col_obj_i,col_trans_i]);
# 
#     
#     RSM/=RSM.max();
#     
#     
#     
#     measure = 0.0;
#     count = 0;
#     for row in range(nObj*nTrans):
#         for col in range(row+1,nObj*nTrans):
#             print(str(row)+' '+str(col));
#             if math.floor(row/nTrans)==math.floor(col/nTrans):
#                 measure+=(1-RSM[row,col]);
# #                 print('plus');
#             else:
#                 measure+=(RSM[row,col]-0);
# #                 print('minus');
#             count+=1;
#             
#     measure/=count;
#     print(1-measure);
#     
#     
#     plt.subplot(1,2,phaseIndex);
#     plt.imshow(RSM,interpolation='none');
#     plt.title(phase + ' (' +str(1-measure)+')');
#     x = range(nObj*nTrans);
#     y = range(nObj*nTrans);
#     plt.xticks(x, labels, rotation='vertical')
#     plt.yticks(y, labels, rotation='horizontal')
# #     print(corr2(RSM,I));
#     phaseIndex+=1;
# fig.savefig("../output/"+experimentName+"/SpikeChain_RSA.png");
# fig.savefig("../output/"+experimentName+"/SpikeChain_RSA.eps");
# plt.show();



