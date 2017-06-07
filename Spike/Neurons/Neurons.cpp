#include "Neurons.hpp"
#include <cassert>
#include <stdlib.h>
#include "../Helpers/TerminalHelpers.hpp"


// Neurons Constructor
Neurons::Neurons() {
	// Variables
	total_number_of_neurons = 0;
	total_number_of_groups = 0;
	number_of_neurons_in_new_group = 0;

	// Host Pointers
	start_neuron_indices_for_each_group = nullptr;
	last_neuron_indices_for_each_group = nullptr;
	per_neuron_afferent_synapse_count = nullptr;
	group_shapes = nullptr;
	per_neuron_efferent_synapse_count = nullptr;
	per_neuron_efferent_synapse_indices = nullptr;
}


// Neurons Destructor
Neurons::~Neurons() {
	free(start_neuron_indices_for_each_group);
	free(last_neuron_indices_for_each_group);
	free(per_neuron_afferent_synapse_count);
	free(group_shapes);
	free(per_neuron_efferent_synapse_count);
	for (int i = 0; i < total_number_of_neurons; i++)
		free(per_neuron_efferent_synapse_indices[i]);
	free(per_neuron_efferent_synapse_indices);
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

	// Extend neuron afferent storage
	per_neuron_efferent_synapse_count = (int*)realloc(per_neuron_efferent_synapse_count, total_number_of_neurons*sizeof(int));
	per_neuron_efferent_synapse_indices = (int**)realloc(per_neuron_efferent_synapse_indices, total_number_of_neurons*sizeof(int*));
	

	// Used for event count
	per_neuron_afferent_synapse_count = (int*)realloc(per_neuron_afferent_synapse_count,(total_number_of_neurons*sizeof(int)));
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		per_neuron_afferent_synapse_count[i] = 0;
		per_neuron_efferent_synapse_count[i] = 0;
		per_neuron_efferent_synapse_indices[i] = nullptr;
	}
	
	return new_group_id;
}

void Neurons::AddEfferentSynapse(int neuron_id, int synapse_id){
	per_neuron_efferent_synapse_count[neuron_id]++;
	per_neuron_efferent_synapse_indices[neuron_id] = (int*)realloc(per_neuron_efferent_synapse_indices[neuron_id], sizeof(int)*per_neuron_efferent_synapse_count[neuron_id]);
	per_neuron_efferent_synapse_indices[neuron_id][per_neuron_efferent_synapse_count[neuron_id] - 1] = synapse_id;
}

void Neurons::reset_state() {
  backend()->reset_state();
}

void Neurons::reset_current_injections() {
  backend()->reset_current_injections();
}

SPIKE_MAKE_STUB_INIT_BACKEND(Neurons);

