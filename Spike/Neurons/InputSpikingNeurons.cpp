#include "InputSpikingNeurons.h"
#include <stdlib.h>
#include <algorithm>
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// InputSpikingNeurons Constructor
InputSpikingNeurons::InputSpikingNeurons() {

	total_number_of_transformations_per_object = 0;
	total_number_of_objects = 0;
	total_number_of_input_stimuli = 0;

	current_stimulus_index = 0;
}


// InputSpikingNeurons Destructor
InputSpikingNeurons::~InputSpikingNeurons() {
}

int InputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	int new_group_id = SpikingNeurons::AddGroup(group_params);
	return (-1*new_group_id - 1);
};


void InputSpikingNeurons::reset_neuron_activities() {

	SpikingNeurons::reset_neuron_activities();

}


bool InputSpikingNeurons::stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index) {
	return true;
}


