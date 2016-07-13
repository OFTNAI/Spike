#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"


enum PRESENTATION_FORMAT {
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI,
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS,
		PRESENTATION_FORMAT_OBJECT_BY_OBJECT_NO_RESET,
		PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS,
		PRESENTATION_FORMAT_RANDOM_NO_RESET
	};

enum OBJECT_ORDER {
	OBJECT_ORDER_RANDOM,
	OBJECT_ORDER_ORIGINAL
};

enum TRANSFORM_ORDER {
	TRANSFORM_ORDER_RANDOM,
	TRANSFORM_ORDER_ORIGINAL
};

struct Stimuli_Presentation_Struct {

	PRESENTATION_FORMAT presentation_format;
	OBJECT_ORDER object_order;
	TRANSFORM_ORDER transform_order;

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

	virtual int* setup_stimuli_presentation_order(Stimuli_Presentation_Struct * stimuli_presentation_params);
	virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);

};


#endif