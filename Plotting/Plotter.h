#ifndef Plotter_H
#define Plotter_H

#include <cuda.h>
#include <curand_kernel.h> // Provides dim3 type. Annoyingly (and strangely) Plotter won't compile without despite curand_kernal not being used specifically.
#include <string>

#include "mgl2/mgl.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"


class Plotter{
public:
	std::string RESULTS_DIRECTORY;

	// Constructor/Destructor
	Plotter(std::string experimentName_param);
	~Plotter();

	void plot_single_cell_information_analysis(SpikeAnalyser * spike_analyser_for_untrained_network, SpikeAnalyser * spike_analyser_for_trained_network);

	void multiple_subplots_test();

};

#endif