#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"


struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};


class InputSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	InputSpikingNeurons();
	~InputSpikingNeurons();

	int total_number_of_input_stimuli;
	int current_stimulus_index;
};


#endif