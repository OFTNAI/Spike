#ifndef Plotter_H
#define Plotter_H

#include <string>

#include "mgl2/mgl.h"

#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"


class Plotter{
public:
	std::string RESULTS_DIRECTORY;

	// Constructor/Destructor
	Plotter(std::string experimentName_param);

	void plot_single_cell_information_analysis(SpikeAnalyser * spike_analyser_for_untrained_network, SpikeAnalyser * spike_analyser_for_trained_network);

	void multiple_subplots_test();

};

#endif
