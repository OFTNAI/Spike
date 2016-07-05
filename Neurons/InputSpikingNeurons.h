#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"

enum STIMULI_PRESENTATION_ORDER_TYPE {
	STIMULI_PRESENTATION_ORDER_TYPE_DEFAULT,
	STIMULI_PRESENTATION_ORDER_TYPE_RANDOM,
	STIMULI_PRESENTATION_ORDER_TYPE_OBJECT_BY_OBJECT_RANDOM_TRANSFORM_ORDER,
	STIMULI_PRESENTATION_ORDER_TYPE_OBJECT_BY_OBJECT_RANDOM_TRANSFORM_ORDER_RESET_BETWEEN_OBJECTS,
};

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};

class InputSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	InputSpikingNeurons();
	~InputSpikingNeurons();

	int total_number_of_input_stimuli;
	int current_stimulus_index;

	virtual int* setup_stimuli_presentation_order(STIMULI_PRESENTATION_ORDER_TYPE stimuli_presentation_order_type);

};


#endif