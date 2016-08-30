import InfoAnalysis
import SpikePlot
# import PolyGroup


#INFO ANALYSIS:
ia = InfoAnalysis.InfoAnalysis()
ia.singleCellInfoAnalysis(['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
ia.singleCellInfoAnalysis(['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = True);


#Struct
# pg = PolyGroup.PolyGroup();
# pg.calcPG(saveImage = True, showImage = False);


#PLOT SPIKES
sp = SpikePlot.SpikePlot();
sp.plotSpikes(['Untrained', 'Trained'],saveImage = True, showImage = False);
# sp.plotSpikes(['Untrained'],saveImage = True, showImage = False);


