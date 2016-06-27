import matplotlib.pyplot as plt #Changed from 'import pylab as plt'
import numpy as np



class SpikePlot(object):
    def plotSpikes(self,saveImage = True, showImage = True):
        
        nLayers = 4;
        exDim = 32;
        inDim = 16;
        
#         fn_id = "../Results/Neurons_Epoch0_SpikeIDs.bin";
#         fn_t = "../Results/Neurons_Epoch0_SpikeTimes.bin";

        fn_id = "../Results/Neurons_SpikeIDs.bin";
        fn_t = "../Results/Neurons_SpikeTimes.bin";
        
        spikeIDs = np.loadtxt(fn_id);
        spikeTimes = np.loadtxt(fn_t);
        
        #fig = plt.figure(0 , figsize=(300, 150),dpi=150);
        fig = plt.figure(0 , figsize=(30, 15),dpi=150);
        
        
        for l in range(nLayers):
            
            cond_ex = (l*exDim*exDim < spikeIDs) & (spikeIDs < (l+1)*exDim*exDim);
            spikeIDs_ex = np.extract((cond_ex), spikeIDs);
            spikeTimes_ex = np.extract(cond_ex, spikeTimes);
                
            cond_in = ((nLayers*exDim*exDim + l*inDim*inDim) < spikeIDs) & (spikeIDs < (nLayers*exDim*exDim + (l+1)*inDim*inDim));
            spikeIDs_in = np.extract(cond_in, spikeIDs);
            spikeTimes_in = np.extract(cond_in, spikeTimes);
            
            spikeIDs_tmp = np.concatenate((spikeIDs_ex-(l*exDim*exDim),spikeIDs_in-((nLayers-1)*exDim*exDim+l*inDim*inDim)));
            spikeTimes_tmp = np.concatenate((spikeTimes_ex,spikeTimes_in));
            
            plt.subplot(nLayers,1,nLayers-l);
            
            plt.plot(spikeTimes_tmp,spikeIDs_tmp,'.',color='k',markersize=2)
        #     plt.plot(spikeTimes_in,spikeIDs_in-((nLayers-1)*exDim*exDim+l*inDim*inDim),'.',color='k',markersize=2)
            
            plt.hold(True)
            
            plt.plot((0,np.max(spikeTimes)),(exDim*exDim,exDim*exDim),'k--', color='r', linewidth=2.0);
            plt.xlabel('time [s]');
            plt.ylabel('cell index (Layer '+str(l) +')');
        #    plt.title('raster plot')
            plt.xlim((0,np.max(spikeTimes)))
            plt.ylim((0,exDim*exDim+inDim*inDim))
        
        if showImage:
            plt.show();
        if saveImage:
            fig.savefig("../Results/rasterPlot.png");
            fig.savefig("../Results/rasterPlot.eps");
            print("figure rasterPlot.png is exported in Results")
