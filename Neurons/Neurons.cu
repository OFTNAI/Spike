

#include "Neurons.h"
#include <stdlib.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// Neurons Constructor
Neurons::Neurons() {

	d_current_injections = NULL;

	// Set totals to zero
	total_number_of_neurons = 0;
	total_number_of_groups = 0;

	// Initialise pointers
	group_shapes = NULL;
	start_neuron_indices_for_each_group = NULL;
	last_neuron_indices_for_each_group = NULL;

}


// Neurons Destructor
Neurons::~Neurons() {

	// Free up memory
	free(group_shapes);
	free(start_neuron_indices_for_each_group);
	free(last_neuron_indices_for_each_group);

}


int Neurons::AddGroup(neuron_parameters_struct * group_params){
	
	number_of_neurons_in_new_group = group_params->group_shape[0] * group_params->group_shape[1];
 
	if (number_of_neurons_in_new_group < 0) {
		print_message_and_exit("Error: Group must have at least 1 neuron.");
	}

	// Update totals
	total_number_of_neurons += number_of_neurons_in_new_group;
	++total_number_of_groups;

	// Calculate new group id
	int new_group_id = total_number_of_groups - 1;

	// Add start neuron index for new group
	start_neuron_indices_for_each_group = (int*)realloc(start_neuron_indices_for_each_group,(total_number_of_groups*sizeof(int)));
	start_neuron_indices_for_each_group[new_group_id] = total_number_of_neurons - number_of_neurons_in_new_group;

	// Add last neuron index for new group
	last_neuron_indices_for_each_group = (int*)realloc(last_neuron_indices_for_each_group,(total_number_of_groups*sizeof(int)));
	last_neuron_indices_for_each_group[new_group_id] = total_number_of_neurons - 1;

	// Add new group shape
	group_shapes = (int**)realloc(group_shapes,(total_number_of_groups*sizeof(int*)));
	group_shapes[new_group_id] = (int*)malloc(2*sizeof(int));
	group_shapes[new_group_id][0] = group_params->group_shape[0];
	group_shapes[new_group_id][1] = group_params->group_shape[1];
	
	return new_group_id;
}


void Neurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps,  bool high_fidelity_spike_storage) {

	CudaSafeCall(cudaMalloc((void **)&d_current_injections, sizeof(float)*total_number_of_neurons));
}

void Neurons::copy_constants_to_device() {

}

void Neurons::reset_neuron_activities() {
	reset_current_injections();
}

void Neurons::reset_current_injections() {
	CudaSafeCall(cudaMemset(d_current_injections, 0.0f, total_number_of_neurons*sizeof(float)));
}


void Neurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

	int number_of_neuron_blocks = (total_number_of_neurons + threads) / threads;
	number_of_neuron_blocks_per_grid.x = number_of_neuron_blocks;
}
