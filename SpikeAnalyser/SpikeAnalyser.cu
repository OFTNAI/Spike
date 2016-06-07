#include "SpikeAnalyser.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"

// SpikeAnalyser Constructor
SpikeAnalyser::SpikeAnalyser(Neurons * neurons_parameter, PoissonSpikingNeurons * input_neurons_parameter) {
	neurons = neurons_parameter;
	input_neurons = input_neurons_parameter;

	per_stimulus_per_neuron_spike_counts = new int*[input_neurons->total_number_of_input_images];

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		per_stimulus_per_neuron_spike_counts[stimulus_index] = new int[neurons->total_number_of_neurons];
	}
	
}


// SpikeAnalyser Destructor
SpikeAnalyser::~SpikeAnalyser() {

}


void SpikeAnalyser::store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus) {
	
	CudaSafeCall(cudaMemcpy(per_stimulus_per_neuron_spike_counts[stimulus_index], 
									d_neuron_spike_counts_for_stimulus, 
									sizeof(float) * neurons->total_number_of_neurons, 
									cudaMemcpyDeviceToHost));

}

void SpikeAnalyser::calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins) {

	int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
	int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
	int number_of_neurons_in_group = neuron_group_end_index - neuron_group_start_index + 1;

	// Find max number of spikes
	int max_number_of_spikes = 0;
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		for (int neuron_index = neuron_group_start_index; neuron_index <= neuron_group_end_index; neuron_index++) {
			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
			if (max_number_of_spikes < number_of_spikes) {
				max_number_of_spikes = number_of_spikes;
			}
		}
	}


	// Calculate bin index for all spike counts + create bin counts for each neuron
	int ** bin_indices_per_stimulus_and_per_neuron = new int*[input_neurons->total_number_of_input_images];
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		bin_indices_per_stimulus_and_per_neuron[stimulus_index] = new int[number_of_neurons_in_group];
	}

	int ** individual_bin_counts_for_each_neuron = new int*[number_of_bins];
	for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
		individual_bin_counts_for_each_neuron[bin_index] = new int[number_of_neurons_in_group];
		for (int neuron_index = 0; neuron_index < number_of_neurons_in_group; neuron_index++) {
			individual_bin_counts_for_each_neuron[bin_index][neuron_index] = 0;
		}
	}


	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {

			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
			float ratio_of_max_number_of_spikes = (float)number_of_spikes / (float)max_number_of_spikes;

			int bin_index = min(int(floor(ratio_of_max_number_of_spikes * number_of_bins)), number_of_bins - 1);
			bin_indices_per_stimulus_and_per_neuron[stimulus_index][neuron_index_zeroed] = bin_index;

			individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed]++;
		}
	}




}