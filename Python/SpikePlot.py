import matplotlib.pyplot as plt #Changed from 'import pylab as plt'
import numpy as np



class SpikePlot(object):
    def loadParams(self,borrowed_globals):
        globals().update(borrowed_globals);
    
    
    def plotSpikes(self,experimentName,phases,saveImage = True, showImage = True, nLayers=4):
        plotAvgFR = True;
        plotSpikes = True;
        
#         nLayers = 4;

#         nObj = 3;
#         nTrans = 2;
#         nLayers = 4;
#         presentationTime = 3.0;
# 
#                 
#         exDim = 64;
#         inDim = 32;
        nExcitCells = exDim*exDim;#32*32;
        nInhibCells = inDim*inDim;#16*16;
        zoomConst = 5;
        
#         fn_id = "../Results/Neurons_Epoch0_SpikeIDs.bin";
#         fn_t = "../Results/Neurons_Epoch0_SpikeTimes.bin";

        if (plotSpikes):
            for phase in phases:
    #             fn_id = "../output/Neurons_SpikeIDs_" + phase + "_Epoch0.txt";
    #             fn_t = "../output/Neurons_SpikeTimes_" + phase + "_Epoch0.txt";
    #             spikeIDs = np.loadtxt(fn_id);
    #             spikeTimes = np.loadtxt(fn_t);
        
                fn_id = "../output/" + experimentName + "/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
                fn_t = "../output/" + experimentName + "/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  
                dtIDs = np.dtype('int32');
                dtTimes = np.dtype('f4');
                
                spikeIDs = np.fromfile(fn_id, dtype=dtIDs);
                spikeTimes = np.fromfile(fn_t, dtype=dtTimes);
    
                cond = spikeTimes>0;
                spikeIDs = np.extract(cond,spikeIDs);
                spikeTimes = np.extract(cond, spikeTimes)
                
                #fig = plt.figure(0 , figsize=(300, 150),dpi=150);
                fig = plt.figure(0 , figsize=(30, 15),dpi=150);
        
        
                     
                               
                plt.plot(spikeTimes,spikeIDs,'.',color='k',markersize=2);
                plt.hold(True)
                for l in range(nLayers):
                    plt.plot((0,np.max(spikeTimes)),(l*(exDim*exDim+inDim*inDim),l*(exDim*exDim+inDim*inDim)),'k-', color='k', linewidth=2.0);
                    plt.plot((0,np.max(spikeTimes)),(l*(exDim*exDim+inDim*inDim)+exDim*exDim,l*(exDim*exDim+inDim*inDim)+exDim*exDim),'k--', color='r', linewidth=2.0);
                
                plt.xlabel('time [s]');
                plt.xlim((0,np.max(spikeTimes)))
                plt.ylim((0,nLayers*(exDim*exDim+inDim*inDim)))
                
                fig.savefig("../output/"+experimentName+"/rasterPlot_"+phase+".png");
        #         fig.savefig("../output/rasterPlot.eps");
                print("figure rasterPlot.png is exported in output")
                plt.clf();
            
        if plotAvgFR:
            for phase in phases:
#                 fig=plt.figure(4 , figsize=(20, 5),dpi=150);
                
                fig=plt.figure(4 , figsize=(zoomConst*2.5*nTrans*nObj, zoomConst*2*nLayers), dpi=1000);
                
                fn_id = "../output/"+experimentName+"/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
                fn_t = "../output/"+experimentName+"/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  
                dtIDs = np.dtype('int32');
                dtTimes = np.dtype('f4');
                
                spikeIDs = np.fromfile(fn_id, dtype=dtIDs);
                spikeTimes = np.fromfile(fn_t, dtype=dtTimes);
                
                cond = spikeTimes>0;
                spikeIDs = np.extract(cond,spikeIDs);
                spikeTimes = np.extract(cond, spikeTimes)
                
                
                #plot Excitatory
                FR = np.zeros((nObj, nTrans,nLayers, nExcitCells));
                for l in range(nLayers):
                    cond_ids = (l*(nExcitCells+nInhibCells) < spikeIDs) & (spikeIDs < l*(nInhibCells+nExcitCells)+nExcitCells);
                    spikeIDs_layer = np.extract(cond_ids, spikeIDs);
                    spikeTimes_layer = np.extract(cond_ids, spikeTimes);
                    for obj in range(nObj):
                        for trans in range(nTrans):
                            cond_stim = ((presentationTime*(obj*nTrans+trans)) < spikeTimes_layer) & (spikeTimes_layer < (presentationTime*(obj*nTrans+trans+1)));
                            spikeIDs_stim = np.extract(cond_stim,spikeIDs_layer)-((nExcitCells+nInhibCells)*l);
                            for id in spikeIDs_stim:
                                FR[obj,trans,l,id]=FR[obj,trans,l,id]+1;
                FR/=presentationTime;
    #             FR = np.random.rand(nObj, nTrans,nLayers, nExcitCells)
            
                for l in range(nLayers):
#                     FRMap_mean = np.zeros((exDim,exDim));
#                     for obj in range(nObj):
#                         for trans in range(nTrans):
#                             for y in range(exDim):
#                                 for x in range(exDim):
#                                     id = x*exDim + y;
#                                     FRMap_mean[y,x] += FR[obj,trans,l,id];
#                     FRMap_mean/=nObj;
#                     FRMap_mean/=nTrans;
                    
                    for obj in range(nObj):
                        for trans in range(nTrans):
                            plt.subplot(nLayers, nTrans*nObj, (nLayers-l-1)*(nTrans*nObj)+(obj*nTrans)+trans+1);
                            plt.title('Firing Rate Map: obj ' + str(obj) )
                            
                            FRMap = np.zeros((exDim,exDim));
                            for y in range(exDim):
                                for x in range(exDim):
                                    id = x*exDim + y;
                                    FRMap[y,x] = FR[obj,trans,l,id];
#                             Rmax = np.max(FRMap);
                            Rmax = 100;
                            plt.imshow(FRMap, cmap='jet', interpolation='none', vmin=0, vmax=Rmax)
                            plt.colorbar();
#                     plt.show();
                fig.savefig("../output/"+experimentName+"/AvgFR_"+phase+".png");
#                 fig.savefig("../output/AvgFR_"+phase+".eps");
                print("figure AvgFR.png is exported in output")
                plt.clf();
                
                
                
                fig=plt.figure(4 , figsize=(zoomConst*2.5*nTrans*nObj, zoomConst*2*nLayers), dpi=1000);
                #plot inhib
                FR = np.zeros((nObj, nTrans,nLayers, nInhibCells));
                for l in range(nLayers):
                    cond_ids = (l*(nExcitCells+nInhibCells)+nExcitCells < spikeIDs) & (spikeIDs < (l+1)*(nInhibCells+nExcitCells));
                    spikeIDs_layer = np.extract(cond_ids, spikeIDs);
                    spikeTimes_layer = np.extract(cond_ids, spikeTimes);
                    for obj in range(nObj):
                        for trans in range(nTrans):
                            cond_stim = ((presentationTime*(obj*nTrans+trans)) < spikeTimes_layer) & (spikeTimes_layer < (presentationTime*(obj*nTrans+trans+1)));
                            spikeIDs_stim = np.extract(cond_stim,spikeIDs_layer)-((nExcitCells+nInhibCells)*l+nExcitCells);
                            for id in spikeIDs_stim:
                                FR[obj,trans,l,id]=FR[obj,trans,l,id]+1;
                FR/=presentationTime;
                
                for l in range(nLayers):
                    for obj in range(nObj):
                        for trans in range(nTrans):
                            plt.subplot(nLayers, nTrans*nObj, (nLayers-l-1)*(nTrans*nObj)+(obj*nTrans)+trans+1);
                            plt.title('Firing Rate Map: obj ' + str(obj) )
                            
                            FRMap = np.zeros((inDim,inDim));
                            for y in range(inDim):
                                for x in range(inDim):
                                    id = x*inDim + y;
                                    FRMap[y,x] = FR[obj,trans,l,id];
#                             Rmax = np.max(FRMap);
                            Rmax = 100;
                            plt.imshow(FRMap, cmap='jet', interpolation='none', vmin=0, vmax=Rmax)
                            plt.colorbar();
#                     plt.show();
                fig.savefig("../output/"+experimentName+"/AvgFR_"+phase+"_inhib.png");
#                 fig.savefig("../output/AvgFR_"+phase+"_inhib.eps");
                print("figure AvgFR_inhib.png is exported in output")
                plt.clf();
            
#         
            
            
        
        
        
        
#         for l in range(nLayers):
#             
#             cond_ex = (l*exDim*exDim < spikeIDs) & (spikeIDs < (l+1)*exDim*exDim);
#             spikeIDs_ex = np.extract((cond_ex), spikeIDs);
#             spikeTimes_ex = np.extract(cond_ex, spikeTimes);
#                 
#             cond_in = ((nLayers*exDim*exDim + l*inDim*inDim) < spikeIDs) & (spikeIDs < (nLayers*exDim*exDim + (l+1)*inDim*inDim));
#             spikeIDs_in = np.extract(cond_in, spikeIDs);
#             spikeTimes_in = np.extract(cond_in, spikeTimes);
#             
#             spikeIDs_tmp = np.concatenate((spikeIDs_ex-(l*exDim*exDim),spikeIDs_in-((nLayers-1)*exDim*exDim+l*inDim*inDim)));
# #             spikeIDs_tmp = np.concatenate((spikeIDs_ex-(l*exDim*exDim),spikeIDs_in-(nLayers*exDim*exDim+l*inDim*inDim)));
#             spikeTimes_tmp = np.concatenate((spikeTimes_ex,spikeTimes_in));
#             
#             plt.subplot(nLayers,1,nLayers-l);
#             
#             plt.plot(spikeTimes_tmp,spikeIDs_tmp,'.',color='k',markersize=2)
#         #     plt.plot(spikeTimes_in,spikeIDs_in-((nLayers-1)*exDim*exDim+l*inDim*inDim),'.',color='k',markersize=2)
#             
#             plt.hold(True)
#             
#             plt.plot((0,np.max(spikeTimes)),(exDim*exDim,exDim*exDim),'k--', color='r', linewidth=2.0);
#             plt.xlabel('time [s]');
#             plt.ylabel('cell index (Layer '+str(l) +')');
#         #    plt.title('raster plot')
#             plt.xlim((0,np.max(spikeTimes)))
#             plt.ylim((0,exDim*exDim+inDim*inDim))
#         
#         if showImage:
#             plt.show();
#         if saveImage:
#             fig.savefig("../output/rasterPlot.png");
#             fig.savefig("../output/rasterPlot.eps");
#             print("figure rasterPlot.png is exported in Results")
