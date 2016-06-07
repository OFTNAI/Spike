#ifndef SpikeAnalyser_H
#define SpikeAnalyser_H

#include <cuda.h>
#include "../Neurons/Neurons.h"
#include "../Neurons/ImagePoissonSpikingNeurons.h"

class SpikeAnalyser{
public:

	// Constructor/Destructor
	SpikeAnalyser(Neurons *neurons_parameter, ImagePoissonSpikingNeurons *input_neurons_parameter);
	~SpikeAnalyser();

	Neurons * neurons;
	ImagePoissonSpikingNeurons * input_neurons;
	
	int number_of_neurons_in_group;

	int ** per_stimulus_per_neuron_spike_counts;

	float ** information_scores_for_each_object_and_neuron;
	float ** descending_information_scores_for_each_object_and_neuron;

	void store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus);

	void calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins);

};

#endif