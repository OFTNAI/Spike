import matplotlib.pyplot as plt #Changed from 'import pylab as plt'
import numpy as np
import itertools
# 
# 
# 
# class PolyGroup(object):
#     def calcPG(self,saveImage = True, showImage = True):
nExcitCells = 32*32;
nInhibCells = 16*16;

nObj = 5;
nTrans = 2;
nLayers = 4;
presentationTime = 1.0;

triggerSize = 3;

dt_int = np.dtype('int32');
dt_float = np.dtype('f4');

#loading network states
fn_preIDs = "../output/Neurons_NetworkPre.bin";
fn_postIDs = "../output/Neurons_NetworkPost.bin";  
fn_weights = "../output/Neurons_NetworkWeights.bin";
fn_delays = "../output/Neurons_NetworkDelays.bin"

preIDs = np.fromfile(fn_preIDs, dtype=dt_int);
postIDs = np.fromfile(fn_postIDs, dtype=dt_int);
weights = np.fromfile(fn_weights, dtype=dt_float);
delays = np.fromfile(fn_delays, dtype=dt_int);


#loading spike trains
phase = "Trained";
fn_id = "../output/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
fn_t = "../output/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  

spikeIDs = np.fromfile(fn_id, dtype=dt_int);
spikeTimes = np.fromfile(fn_t, dtype=dt_float);
    

#remove input neurons:
cond = preIDs>=0;
preIDs = np.extract(cond, preIDs);
postIDs = np.extract(cond, postIDs);
weights = np.extract(cond, weights);
delays = np.extract(cond, delays);
connectionIndex = np.arange(delays.size);

maxWeights = np.max(weights);
weightTh = 0.7;

#structure of index:
# excitCells in 1st layer, inhibCells in 1st layer, excitCells in 2nd layer, ...

firstBegin = 0;
firstEnd = nExcitCells-1;
secondBegin = nExcitCells + nInhibCells;
secondEnd = nExcitCells*2 + nInhibCells - 1;

for i_post in range(secondBegin,secondEnd+1):
    cond_post = postIDs == i_post;
    cond_pre = preIDs<=firstEnd;
    cond_weights = weights>maxWeights*weightTh;
    cond_comb = cond_post & cond_pre & cond_weights;
    connectionIndex_preSyn = np.extract(cond_comb,connectionIndex);
    if connectionIndex_preSyn.size > triggerSize:
        for triggers in np.array(list(itertools.combinations(connectionIndex_preSyn,triggerSize))):
            maxDelay = delays[triggers].max();
            i_pre_list = preIDs[triggers];
            w_list[triggers];
            
                    
                    
                    
                    
                    
                    
                
                
            
            
        