#include "GraphPlotter.h"

#include <stdlib.h>

#include "../Helpers/matplotlib-cpp-master/matplotlibcpp.h"
#include <cmath>
namespace plt = matplotlibcpp;

// GraphPlotter Constructor
GraphPlotter::GraphPlotter() {

	
}


// GraphPlotter Destructor
GraphPlotter::~GraphPlotter() {

}


void GraphPlotter::plot_untrained_vs_trained_single_cell_information_for_all_objects(SpikeAnalyser *spike_analyser_for_untrained_network, SpikeAnalyser *spike_analyser_for_trained_network) {
	// Prepare data.
	int n = spike_analyser_for_untrained_network->number_of_neurons_in_group;

	




	// for 
	std::vector<float> x(n), untrained(n), trained(n);
	for(int i=0; i<n; ++i) {
		x.at(i) = i;
		untrained.at(i) = spike_analyser_for_untrained_network->descending_information_scores_for_each_object_and_neuron[0][i];
		trained.at(i) = spike_analyser_for_trained_network->descending_information_scores_for_each_object_and_neuron[0][i];
	}

	plt::named_plot("Untrained", x, untrained, "r--");
	plt::named_plot("Trained", x, trained, "b--");

	// // Plot line from given x and y data. Color is selected automatically.
	// plt::plot(x, y);
	// // Plot a red dashed line from given x and y data.
	// plt::plot(x, w,"r--");
	// Plot a line whose name will show up as "log(x)" in the legend.
	// plt::named_plot("log(x)", x, z);

	// Set x-axis to interval [0,1000000]
	plt::xlim(0, n);

	// Add graph title
	plt::title("Sample figure");
	// Enable legend.
	plt::legend();
	// save figure
	plt::save("./basic.png");
}


