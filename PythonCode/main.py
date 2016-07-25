import InfoAnalysis
import SpikePlot
import PolyGroup


#INFO ANALYSIS:
# ia = InfoAnalysis.InfoAnalysis()
# ia.singleCellInfoAnalysis(['Neurons_Epoch0_t25077681', 'Neurons_Epoch0_t74915752'],weightedAnalysis = 1,saveImage = True, showImage = True);
# ia.singleCellInfoAnalysis(['Neurons_Epoch0_'],weightedAnalysis = 1,saveImage = True, showImage = False);


#Struct
# pg = PolyGroup.PolyGroup();
# pg.polyGroup(saveImage = True, showImage = False);


#PLOT SPIKES
sp = SpikePlot.SpikePlot();
sp.plotSpikes(['Neurons_Epoch0_t62631290'],saveImage = True, showImage = False);


