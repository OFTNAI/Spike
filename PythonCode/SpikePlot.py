import matplotlib.pyplot as plt #Changed from 'import pylab as plt'
import numpy as np



class SpikePlot(object):
    def plotSpikes(self,phases,saveImage = True, showImage = True):
        
        nLayers = 4;
        exDim = 32;
        inDim = 16;
        
#         fn_id = "../Results/Neurons_Epoch0_SpikeIDs.bin";
#         fn_t = "../Results/Neurons_Epoch0_SpikeTimes.bin";

        for phase in phases:
#             fn_id = "../output/Neurons_SpikeIDs_" + phase + "_Epoch0.txt";
#             fn_t = "../output/Neurons_SpikeTimes_" + phase + "_Epoch0.txt";
#             spikeIDs = np.loadtxt(fn_id);
#             spikeTimes = np.loadtxt(fn_t);
    
            fn_id = "../output/Neurons_SpikeIDs_" + phase + "_Epoch0.bin";
            fn_t = "../output/Neurons_SpikeTimes_" + phase + "_Epoch0.bin";  
            dtIDs = np.dtype('int32');
            dtTimes = np.dtype('f4');
            
            spikeIDs = np.fromfile(fn_id, dtype=dtIDs);
            spikeTimes = np.fromfile(fn_t, dtype=dtTimes);

            
            #fig = plt.figure(0 , figsize=(300, 150),dpi=150);
            fig = plt.figure(0 , figsize=(30, 15),dpi=150);
    
            plt.plot(spikeTimes,spikeIDs,'.',color='k',markersize=2);
            for l in range(nLayers):
                plt.plot((0,np.max(spikeTimes)),(l*(exDim*exDim+inDim*inDim),l*(exDim*exDim+inDim*inDim)),'k-', color='k', linewidth=2.0);
                plt.plot((0,np.max(spikeTimes)),(l*(exDim*exDim+inDim*inDim)+exDim*exDim,l*(exDim*exDim+inDim*inDim)+exDim*exDim),'k--', color='r', linewidth=2.0);
            
            plt.xlabel('time [s]');
            plt.xlim((0,np.max(spikeTimes)))
            plt.ylim((0,nLayers*(exDim*exDim+inDim*inDim)))
            
            fig.savefig("../output/rasterPlot_"+phase+".png");
    #         fig.savefig("../output/rasterPlot.eps");
            print("figure rasterPlot.png is exported in Results")
            plt.hold(True)
            
            
        
        
        
        
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
