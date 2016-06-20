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

	int number_of_neurons_in_group = spike_analyser_for_untrained_network->number_of_neurons_in_group;
	
	for (int object_index = 0; object_index < spike_analyser_for_untrained_network->input_neurons->total_number_of_objects; object_index++) {

		std::vector<float> neuron_indices(number_of_neurons_in_group);

		std::vector<float> untrained_descending_scores = std::vector<float>(spike_analyser_for_untrained_network->descending_information_scores_for_each_object_and_neuron[object_index], spike_analyser_for_untrained_network->descending_information_scores_for_each_object_and_neuron[object_index] + number_of_neurons_in_group);
		std::vector<float> trained_descending_scores = std::vector<float>(spike_analyser_for_trained_network->descending_information_scores_for_each_object_and_neuron[object_index], spike_analyser_for_trained_network->descending_information_scores_for_each_object_and_neuron[object_index] + number_of_neurons_in_group);
		
		for(int neuron_index=0; neuron_index < number_of_neurons_in_group; neuron_index++) {

			neuron_indices.at(neuron_index) = neuron_index;
		}

		plt::named_plot("Untrained", neuron_indices, untrained_descending_scores, "--");
		plt::named_plot("Trained", neuron_indices, trained_descending_scores);

	}


	// Set x-axis
	plt::xlim(0, number_of_neurons_in_group);

	// Add graph title
	plt::title("Single Cell Information Anlysis Scores");
	// Enable legend.
	plt::legend();
	// save figure
	plt::save("./SingleCellInformationAnlysisScores.png");
}

void GraphPlotter::plot_all_spikes(RecordingElectrodes * recording_electrodes) {

	plt::figure();

	// std::vector<float>spike_times = std::vector<float>(recording_electrodes->h_time_in_seconds_of_stored_spikes_on_host, recording_electrodes->h_time_in_seconds_of_stored_spikes_on_host + recording_electrodes->h_total_number_of_spikes_stored_on_host);

	// std::vector<float> spike_neuron_ids(recording_electrodes->h_total_number_of_spikes_stored_on_host);
	// printf("h_total_number_of_spikes_stored_on_host: %d\n", recording_electrodes->h_total_number_of_spikes_stored_on_host);
	// for(int spike_index=0; spike_index < recording_electrodes->h_total_number_of_spikes_stored_on_host; spike_index++) {
	// 	spike_neuron_ids.at(spike_index) = (float)recording_electrodes->h_neuron_ids_of_stored_spikes_on_host[spike_index];
	// }

	// // Set x-axis
	// plt::xlim(0.0, 0.5);

	// // std::map< std::string, std::string> plot_map;
	// // plot_map.insert(std::make_pair("marker", "o"));
	// // plt::plot(spike_times,spike_neuron_ids, plot_map);
	// // plt::plot(spike_times, spike_neuron_ids);
	// plt::named_plot("HEY", spike_times,spike_neuron_ids, ".");

	// // Enable legend.
	// plt::legend();

	// plt::save("./Results/rasterPlot.png");


	// // Set x-axis
	// plt::xlim(0.0, 0.5);

	// plt::named_plot("HEY2", spike_times,spike_neuron_ids, ".");

	// // Enable legend.
	// plt::legend();

	// plt::save("./Results/rasterPlot2.png");


}
