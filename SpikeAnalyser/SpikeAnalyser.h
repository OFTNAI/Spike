#ifndef SpikeAnalyser_H
#define SpikeAnalyser_H

#include <cuda.h>

class SpikeAnalyser{
public:

	// Constructor/Destructor
	SpikeAnalyser();
	~SpikeAnalyser();

	void store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus);

};

#endif