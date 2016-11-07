import InfoAnalysis
import SpikePlot
# import PolyGroup


nObj = 3;
nTrans = 2;
nLayers = 4;
presentationTime = 2.0;
exDim = 64;
inDim = 32;



#INFO ANALYSIS:
ia = InfoAnalysis.InfoAnalysis()
ia.loadParams(globals());
ia.singleCellInfoAnalysis(['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
ia.singleCellInfoAnalysis(['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = True);

# ia.singleCellInfoAnalysis(['Untrained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
# ia.singleCellInfoAnalysis(['Untrained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = True);

# ia.singleCellInfoAnalysis(['Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
# ia.singleCellInfoAnalysis(['Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=3,plotAllSingleCellInfo = True);

#Struct
# pg = PolyGroup.PolyGroup();
# pg.calcPG(saveImage = True, showImage = False);


#PLOT SPIKES
sp = SpikePlot.SpikePlot();
sp.loadParams(globals());
# sp.plotSpikes(['Untrained', 'Trained'],saveImage = True, showImage = False);
# sp.plotSpikes(['Trained'],saveImage = True, showImage = False, nLayers = 4);
# sp.plotSpikes(['Untrained'],saveImage = True, showImage = False, nLayers = 4);
