#ifndef SpikingModel_H
#define SpikingModel_H

#include <cuda.h>
#include <stdio.h>
#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>
#include "../Plotting/Plotter.h"
#include <vector>

#include <iostream>
using namespace std;


class SpikingModel {

public:

	// Constructor/Destructor
	SpikingModel();
	~SpikingModel();


	SpikingNeurons * neurons;
	SpikingSynapses * synapses;
	InputSpikingNeurons * input_neurons;
	STDP* stdp_rule; 


};

#endif