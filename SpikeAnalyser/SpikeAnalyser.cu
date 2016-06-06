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

	printf("start: %d, end: %d\n", neuron_group_start_index, neuron_group_end_index);

	int max_number_of_spikes = 0;

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		for (int neuron_index = 0; neuron_index < neurons->total_number_of_neurons; neuron_index++) {

			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
			if (max_number_of_spikes < number_of_spikes) {
				max_number_of_spikes = number_of_spikes;
			}

			// printf("max_number_of_spikes: %d\n", max_number_of_spikes);
			// printf("per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]: %d\n", per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]);
		}
	}
}