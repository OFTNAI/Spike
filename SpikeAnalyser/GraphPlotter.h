#ifndef GraphPlotter_H
#define GraphPlotter_H

#include <cuda.h>
#include "SpikeAnalyser.h"

class GraphPlotter{
public:

	// Constructor/Destructor
	GraphPlotter();
	~GraphPlotter();

	void plot_untrained_vs_trained_single_cell_information_for_all_objects(SpikeAnalyser *spike_analyser_for_untrained_network, SpikeAnalyser *spike_analyser_for_trained_network);

};

#endif