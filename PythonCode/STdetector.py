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

experimentName = '20160904_FF_successful';
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
maxDelay = 5;

useMaxFRTh=True;
maxFRTh = 100;

saveImage = True;
showImage = False;
nBins=5;
weightedAnalysis = False;
plotAllSingleCellInfo = True;

fig=plt.figure(4 , figsize=(20, 5),dpi=150);
phaseIndex=1;
for phase in phases:
    maxSCindex=np.zeros((nObj,nTrans,4,nExcitCells));
    probTable_list = np.zeros((nObj,nTrans,nExcitCells,nExcitCells,maxDelay+1));#targetcell, 
    
    l = 3;
    if os.path.isfile("../output/"+experimentName+"/maxScindex_" + phase + "_Epoch0.bin"):
        maxSCindex = pickle.load( open( "../output/"+experimentName+"/maxScindex_" + phase + "_Epoch0.bin", "rb" ) )
    else:
        fn_id = "../output/"+experimentName+"/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
        fn_t = "../output/"+experimentName+"/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  
        dtIDs = np.dtype('int32');
        dtTimes = np.dtype('f4');
         
        spikeIDs = np.fromfile(fn_id, dtype=dtIDs);
        spikeTimes = (np.fromfile(fn_t, dtype=dtTimes)*1000).astype(int);
         
        cond_ids = (l*(nExcitCells+nInhibCells) < spikeIDs) & (spikeIDs < l*(nInhibCells+nExcitCells)+nExcitCells) & (spikeTimes<presentationTime*nObj*nTrans);
        spikeIDs_layer = np.extract(cond_ids, spikeIDs)-l*(nExcitCells+nInhibCells);
        spikeTimes_layer = np.extract(cond_ids, spikeTimes);
         
        FiringTable = [];
        for t in range(presentationTime*nObj*nTrans):
            FiringTable.append([]);
         
        for index in range(spikeIDs_layer.size):
            FiringTable[int(spikeTimes_layer[index])].append(spikeIDs_layer[index]);
         
         
         
        for obj in range(nObj):
            for trans in range(nTrans):
#         for obj in range(1):
#             for trans in range(2):
                ProbTable = np.zeros((nExcitCells,nExcitCells,maxDelay+1));#targetcell, 
                print(str(obj) + ',' + str(trans));
                cond_stim = ((presentationTime*(obj*nTrans+trans)) < spikeTimes_layer) & (spikeTimes_layer < (presentationTime*(obj*nTrans+trans+1)));
                spikeIDs_stim = np.extract(cond_stim,spikeIDs_layer);
                spikeTimes_stim = np.extract(cond_stim,spikeTimes_layer);
                num_firing = 0;
                for index_spike in range(spikeIDs_stim.size):
                    targetSpikeIndex = spikeIDs_stim[index_spike];
                    t_begin = max(presentationTime*(obj*nTrans+trans),spikeTimes_stim[index_spike]-maxDelay);
                    num_firing+=len(FiringTable[t]);
                     
                    for t in range(t_begin,spikeTimes_stim[index_spike]):
                        ProbTable[targetSpikeIndex,FiringTable[t],t-t_begin]+=1;
                        probTable_list[obj,trans,targetSpikeIndex,FiringTable[t],t-t_begin]+=1;
                 
                if (num_firing>0):
                    ProbTable/=num_firing;
                    probTable_list[obj,trans]/=num_firing;
                 
    #             plt.figure();
                colorBarOn = False;
                for cell in range(nExcitCells):
                    maxSCindex[obj,trans,l,cell] = np.amax(ProbTable[cell]);
    #                 if(np.amax(ProbTable[cell])>0):
    #                     plt.imshow(ProbTable[cell], extent=[0,maxDelay,0,nExcitCells], aspect='auto')
    #                     if(not colorBarOn):
    #                         plt.colorbar();
    #                     plt.title(str(cell))
    #                     plt.show();
         
#         pickle.dump(maxSCindex, open( "../output/"+experimentName+"/maxScindex_" + phase + "_Epoch0.bin", "wb" ));
#         pickle.dump(probTable_list, open( "../output/"+experimentName+"/probTableList_" + phase + "_Epoch0.bin", "wb" ));
     
    RSM = np.zeros((nObj*nTrans,nObj*nTrans));
    
    probTable_list/=probTable_list.max();
#     for row in range(nExcitCells*nObj*nTrans):#row = obj*trans+i
#         print(str(round(row*100.0/(nExcitCells*nObj*nTrans))));
#         row_cell_i = row%(nExcitCells);
#         row_obj_i = math.floor(1.0*row/(nExcitCells*nTrans))%nObj;
#         row_trans_i = math.floor(1.0*row/nExcitCells)%nTrans;
# #         print('row_cell_i:'+str(row_cell_i)+', row_obj_i:'+str(row_obj_i)+', row_trans_i:'+str(row_trans_i));
#         if (probTable_list[row_obj_i,row_trans_i,row_cell_i].max()==0):
#             continue;
#         for col in range(nExcitCells*nObj*nTrans):
#             col_cell_i = col%(nExcitCells);
#             col_obj_i = math.floor(1.0*col/(nExcitCells*nTrans))%nObj;
#             col_trans_i = math.floor(1.0*col/nExcitCells)%nTrans;
# #             print('\trow_cell_i:'+str(col_cell_i)+', row_obj_i:'+str(col_obj_i)+', row_trans_i:'+str(col_trans_i));
#             if(probTable_list[col_obj_i,col_trans_i,col_cell_i].max()==0):
#                 continue;
# #             print(corr2(np.reshape(probTable_list[row_obj_i,row_trans_i,row_cell_i], (nExcitCells,maxDelay+1)),np.reshape(probTable_list[col_obj_i,col_trans_i,col_cell_i], (nExcitCells,maxDelay+1))));
#             RSM[int(row_obj_i*nTrans+row_trans_i),int(col_obj_i*nTrans+col_trans_i)]+=corr2(probTable_list[row_obj_i,row_trans_i,row_cell_i],probTable_list[col_obj_i,col_trans_i,col_cell_i]);
    
    labels=[];
    for obj in range(nObj):
        for trans in range(nTrans):
            labels.append('obj:'+str(obj)+', trans:'+str(trans));
    
    for row in range(nObj*nTrans):#row = obj*trans+i
        print(str(round(row*100.0/(nObj*nTrans))));
        row_trans_i = row%nTrans;
        row_obj_i = math.floor(1.0*row/nTrans)%nObj;
#         print('row_cell_i:'+str(row_cell_i)+', row_obj_i:'+str(row_obj_i)+', row_trans_i:'+str(row_trans_i));
        if (probTable_list[row_obj_i,row_trans_i].max()==0):
            continue;
        for col in range(nObj*nTrans):
            col_trans_i = col%nTrans;
            col_obj_i = math.floor(1.0*col/nTrans)%nObj;
#             print('\trow_cell_i:'+str(col_cell_i)+', row_obj_i:'+str(col_obj_i)+', row_trans_i:'+str(col_trans_i));
            if(probTable_list[col_obj_i,col_trans_i].max()==0):
                continue;
#             print(corr2(np.reshape(probTable_list[row_obj_i,row_trans_i,row_cell_i], (nExcitCells,maxDelay+1)),np.reshape(probTable_list[col_obj_i,col_trans_i,col_cell_i], (nExcitCells,maxDelay+1))));
            RSM[int(row_obj_i*nTrans+row_trans_i),int(col_obj_i*nTrans+col_trans_i)]+=corr2(probTable_list[row_obj_i,row_trans_i],probTable_list[col_obj_i,col_trans_i]);

    
    RSM/=RSM.max();
    plt.subplot(1,2,phaseIndex);
    plt.imshow(RSM,interpolation='none');
    plt.title(phase);
    x = range(nObj*nTrans);
    y = range(nObj*nTrans);
    plt.xticks(x, labels, rotation='vertical')
    plt.yticks(y, labels, rotation='horizontal')
    
#     performanceMeasure = 0.0;
#     
#     if(not useMaxFRTh):
#         maxFRTh = maxSCindex[:,:,l,:].max()
#     print(" Maximum Firing Rate Threshold of " + str(maxFRTh) +" is used");
#     
#     if maxSCindex[:,:,l,:].max()!=0:
#         FR_tmp = maxSCindex/maxFRTh;
#     else:
#         FR_tmp = maxSCindex;
#         
#         
#     infos = np.zeros(nExcitCells);
#     
#     sumPerBin = np.zeros((nExcitCells,nBins));
#     sumPerObj = nTrans;
#     sumPerCell = nTrans*nObj;
#     IRs = np.zeros((nObj,nExcitCells));#I(R,s) single cell information
#     IRs_weighted = np.zeros((nObj,nExcitCells));#I(R,s) single cell information
#     pq_r = np.zeros(nObj)#prob(s') temporally used in the decoing process
#     Ps = 1/nObj   #Prob(s) 
#     
# 
#     
#     print("**Loading data**")
#     binMatrix = np.zeros((nExcitCells, nObj, nBins));# #number of times when fr is classified into a specific bin within a specific objs's transformations
#     for obj in range(nObj):
#         print str(obj) + '/' + str(nObj);
#         for trans in range(nTrans):
#             for cell in range(nExcitCells):
# #                             bin = np.around(FR_tmp[obj,trans,l,cell]*(nBins-1));
#                 bin = min(np.floor((FR_tmp[obj,trans,l,cell])*(nBins)),nBins-1)
#                 binMatrix[cell,obj,bin]=binMatrix[cell,obj,bin]+1;
# 
#     
#     print "** single-cell information analysis **";
#     # Loop through all cells to calculate single cell information
#     for cell in range(nExcitCells):
#         # For each cell, count the number of transforms per bin
#         for bin in range(nBins):
#             for obj in range(nObj):
#                 sumPerBin[cell,bin]+=binMatrix[cell,obj,bin];
# 
#         # Calculate the information for cell_x cell_y per stimulus
#         for obj in range(nObj):
#             for bin in range(nBins):
#                 Pr = sumPerBin[cell,bin]/sumPerCell;
#                 Prs = binMatrix[cell,obj,bin]/sumPerObj;
#                 if(Pr!=0 and Prs!=0 and Pr<Prs):
#                     IRs[obj,cell]+=(Prs*(np.log2(Prs/Pr)));#*((bin-1)/(nBins-1)); #could be added to weight the degree of firing rates.
#                     #IRs(row,col,obj)=IRs(row,col,obj)+(Prs*(log2(Prs/Pr)))*((bin-1)/(nBins-1)); #could be added to weight the degree of firing rates.
#                     IRs_weighted[obj,cell]+=(Prs*(np.log2(Prs/Pr)))*((bin)/(nBins-1)); #could be added to weight the degree of firing rates.
#  
#     if (weightedAnalysis):
#         IRs = IRs_weighted;
#     
#     
#     if (plotAllSingleCellInfo):
#         IRs_sorted = np.sort(IRs*-1)*-1;
# #         plt.subplot(2,nLayers,(phaseIndex-1)*nLayers+(l+1));
#         plt.subplot(1,2,phaseIndex);
# 
# #                     plt.plot(np.transpose(IRs_sorted), color='k');
#         plt.plot(np.transpose(IRs_sorted));
#         
#         plt.ylim([-0.05, np.log2(nObj)+0.05]);
#         plt.xlim([0, nExcitCells])
#         plt.title("Layer " + str(l));
#         plt.ylabel("Information [bit]");
#         plt.xlabel("Cell Rank");
#         
#     else:
#         IRs = np.max(IRs,axis=0);
#         
#         IRs_sorted = np.transpose(np.sort(IRs));
#         reversed_arr = IRs_sorted[::-1]
#     
#     
#         infos = reversed_arr;
#     
# #         plt.subplot(1,nLayers,l+1);
# 
#         if phaseIndex==1:
#             plt.plot(np.transpose(infos), linestyle='--', color='k');
#         elif phaseIndex==2:
#             plt.plot(np.transpose(infos), linestyle='-', color='k');
#         plt.ylim([-0.05, np.log2(nObj)+0.05]);
#         plt.xlim([0, nExcitCells])
#         plt.title("Layer " + str(l));
#         plt.ylabel("Information [bit]");
#         plt.xlabel("Cell Rank");
#         plt.hold( );
                
        

#         plt.savefig(os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/singleCellInfo_bin"+str(nBins)+".png");
#         plt.savefig(os.path.split(os.path.realpath(__file__))[0] +"/Results/"+experimentName+"/singleCellInfo_bin"+str(nBins)+".eps");
    phaseIndex+=1;
plt.show();
# if showImage:
#     plt.show();
# if saveImage:
#     if (plotAllSingleCellInfo):
#         fig.savefig("../output/"+experimentName+"/Chain_SingleCellInfo_ALL.png");
# #                 fig.savefig("../output/SingleCellInfo_ALL.eps");
#     else:
#           fig.savefig("../output/"+experimentName+"/Chain_SingleCellInfo_MAX.png");
# #                   fig.savefig("../output/SingleCellInfo_MAX.eps");              
# 
#     print("figure SingleCellInfo.png is exported in Results") 
# 
# plt.close();



