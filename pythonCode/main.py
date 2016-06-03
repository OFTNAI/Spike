import InfoAnalysis
import spikePlot


#INFO ANALYSIS:
# ia = InfoAnalysis.InfoAnalysis(globals())
ia = InfoAnalysis.InfoAnalysis()
ia.singleCellInfoAnalysis(['Neurons_Epoch0'],weightedAnalysis = 1,saveImage = True, showImage = True);

#PLOT SPIKES
sp = spikePlot.SpikePlot();
sp.plotSpikes(saveImage = True, showImage = False);