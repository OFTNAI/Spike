#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

/*CUDA
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
*/

#include "SpikingNeurons.hpp"

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {

};

class InputSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	InputSpikingNeurons();
	~InputSpikingNeurons();

	
	int current_stimulus_index;

	int total_number_of_input_stimuli;
	int total_number_of_objects;
	int total_number_of_transformations_per_object;

	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void reset_neuron_activities();
	virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);

};


#endif
