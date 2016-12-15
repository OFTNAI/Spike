import InfoAnalysis
import SpikePlot
# import PolyGroup


nObj = 2;
nTrans = 4;
nLayers = 4;
presentationTime = 2.0;
exDim = 64;
inDim = 32;


# experimentName = '20160904_FF_successful';
# experimentName = '20160908_FF_LAT';
# experimentName = '20160908_FF_FB';
experimentName = '1.1e--nCon2_useDiffInputsForTrainTest_10ep';
# experimentName = 'test';

#INFO ANALYSIS:
ia = InfoAnalysis.InfoAnalysis()
ia.loadParams(globals());

# ia.singleCellInfoAnalysis(experimentName,['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=3,plotAllSingleCellInfo = False);


# ia.singleCellInfoAnalysis(experimentName,['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
# ia.singleCellInfoAnalysis(experimentName,['Untrained', 'Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = True);

# ia.singleCellInfoAnalysis(experimentName,['Untrained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
# ia.singleCellInfoAnalysis(experimentName,['Untrained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = True);

# ia.singleCellInfoAnalysis(experimentName,['Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=5,plotAllSingleCellInfo = False);
# ia.singleCellInfoAnalysis(experimentName,['Trained'],weightedAnalysis = False, saveImage = True, showImage = False, nBins=3,plotAllSingleCellInfo = True);

#Struct
# pg = PolyGroup.PolyGroup();
# pg.calcPG(saveImage = True, showImage = False);


#PLOT SPIKES
sp = SpikePlot.SpikePlot();
sp.loadParams(globals());
sp.plotSpikes(experimentName,['Untrained', 'Trained'],saveImage = True, showImage = False);
# sp.plotSpikes(experimentName,['Trained'],saveImage = True, showImage = False, nLayers = 4);
# sp.plotSpikes(experimentName,['Untrained'],saveImage = True, showImage = False, nLayers = 4);


