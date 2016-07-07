#include "InputSpikingNeurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// InputSpikingNeurons Constructor
InputSpikingNeurons::InputSpikingNeurons() {
	total_number_of_input_stimuli = 0;
	current_stimulus_index = 0;
}


// InputSpikingNeurons Destructor
InputSpikingNeurons::~InputSpikingNeurons() {
}


int* InputSpikingNeurons::setup_stimuli_presentation_order(Stimuli_Presentation_Struct * stimuli_presentation_params) {
	
	int* stimuli_presentation_order = (int*)malloc(total_number_of_input_stimuli*sizeof(int));
	
	for (int i = 0; i < total_number_of_input_stimuli; i++){
		stimuli_presentation_order[i] = i;
	}

	switch (stimuli_presentation_params->presentation_format) {

		case PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS: case PRESENTATION_FORMAT_RANDOM_NO_RESET: {
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[total_number_of_input_stimuli]);
			break;
		}

		default:
			break;
	}

	return stimuli_presentation_order;
}

bool InputSpikingNeurons::stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index) {
	print_message_and_exit("Object by object presentation currently unsupported at InputSpikingNeurons level. Please use ImagePoissonInputSpikingNeurons.");
	return false;
}

void InputSpikingNeurons::reset_neurons() {

	SpikingNeurons::reset_neurons();

}
