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