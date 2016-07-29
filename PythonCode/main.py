import InfoAnalysis
import SpikePlot
import PolyGroup


#INFO ANALYSIS:
ia = InfoAnalysis.InfoAnalysis()
ia.singleCellInfoAnalysis(['Untrained', 'Trained'],weightedAnalysis = True,saveImage = True, showImage = False);


#Struct
# pg = PolyGroup.PolyGroup();
# pg.polyGroup(saveImage = True, showImage = False);


#PLOT SPIKES
# sp = SpikePlot.SpikePlot();
# sp.plotSpikes(['Untrained'],saveImage = True, showImage = False);
# sp.plotSpikes(['Untrained', 'Trained'],saveImage = True, showImage = False);


