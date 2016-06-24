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