#ifndef SpikeAnalyser_H
#define SpikeAnalyser_H

#include <cuda.h>
#include "../Neurons/Neurons.h"
#include "../Neurons/PoissonSpikingNeurons.h"

class SpikeAnalyser{
public:

	// Constructor/Destructor
	SpikeAnalyser(Neurons *neurons_parameter, PoissonSpikingNeurons *input_neurons_parameter);
	~SpikeAnalyser();

	Neurons * neurons;
	PoissonSpikingNeurons * input_neurons;

	int ** per_stimulus_per_neuron_spike_counts;

	void store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus);

};

#endif