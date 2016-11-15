#include "Neurons.hpp"
#include <stdlib.h>
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.hpp"
#include "../Helpers/TerminalHelpers.hpp"


// Neurons Constructor
Neurons::Neurons() {

	// Variables
	total_number_of_neurons = 0;
	total_number_of_groups = 0;
	number_of_neurons_in_new_group = 0;

	// Host Pointers
	start_neuron_indices_for_each_group = NULL;
	last_neuron_indices_for_each_group = NULL;
	per_neuron_afferent_synapse_count = NULL;
	group_shapes = NULL;

	// Device Pointers
	d_per_neuron_afferent_synapse_count = NULL;
	d_current_injections = NULL;

}


// Neurons Destructor
Neurons::~Neurons() {

	free(start_neuron_indices_for_each_group);
	free(last_neuron_indices_for_each_group);
	free(per_neuron_afferent_synapse_count);
	free(group_shapes);

        /*CUDA
	CudaSafeCall(cudaFree(d_per_neuron_afferent_synapse_count));
	CudaSafeCall(cudaFree(d_current_injections));
        */

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

	// Used for event count
	per_neuron_afferent_synapse_count = (int*)realloc(per_neuron_afferent_synapse_count,(total_number_of_neurons*sizeof(int)));
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		per_neuron_afferent_synapse_count[i] = 0;
	}

	
	return new_group_id;
}


void Neurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps,  bool high_fidelity_spike_storage) {

  /*CUDA
	CudaSafeCall(cudaMalloc((void **)&d_current_injections, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_per_neuron_afferent_synapse_count, sizeof(int)*total_number_of_neurons));
  */
}

void Neurons::copy_constants_to_device() {
  //CUDA CudaSafeCall(cudaMemcpy(d_per_neuron_afferent_synapse_count, per_neuron_afferent_synapse_count, sizeof(int)*total_number_of_neurons, cudaMemcpyHostToDevice));
}

void Neurons::reset_neuron_activities() {
  reset_current_injections();
}

void Neurons::reset_current_injections() {
  //CUDA CudaSafeCall(cudaMemset(d_current_injections, 0.0f, total_number_of_neurons*sizeof(float)));
}


void Neurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

	int number_of_neuron_blocks = (total_number_of_neurons + threads) / threads;
	number_of_neuron_blocks_per_grid.x = number_of_neuron_blocks;
}
