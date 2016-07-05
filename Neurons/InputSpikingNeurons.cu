#include "InputSpikingNeurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// InputSpikingNeurons Constructor
InputSpikingNeurons::InputSpikingNeurons() {
	total_number_of_input_stimuli = 0;
	current_stimulus_index = 0;
}


// InputSpikingNeurons Destructor
InputSpikingNeurons::~InputSpikingNeurons() {
}


int* InputSpikingNeurons::setup_stimuli_presentation_order(STIMULI_PRESENTATION_ORDER_TYPE stimuli_presentation_order_type) {
	
	int* stimuli_presentation_order = (int*)malloc(total_number_of_input_stimuli*sizeof(int));
	
	switch (stimuli_presentation_order_type)
	{
		case STIMULI_PRESENTATION_ORDER_TYPE_DEFAULT:
			for (int i = 0; i < total_number_of_input_stimuli; i++){
				stimuli_presentation_order[i] = i;
			}
			break;

		case STIMULI_PRESENTATION_ORDER_TYPE_RANDOM:
			for (int i = 0; i < total_number_of_input_stimuli; i++){
				stimuli_presentation_order[i] = i;
			}
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[total_number_of_input_stimuli]);
			break;

		default:
			break;

	}

	return stimuli_presentation_order;
}