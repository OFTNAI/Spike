//	Neurons Class C++
//	Neurons.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015
//
//  Adapted from NeuronPopulations by Nasir Ahmad and James Isbister
//	Date: 6/4/2016

#include "Neurons.h"
#include <stdlib.h>
#include <stdio.h>


// Neurons Constructor
Neurons::Neurons() {

	// Set totals to zero
	total_number_of_neurons = 0;
	total_number_of_groups = 0;

	// Initialise pointers
	group_shapes = NULL;
	group_parameters = NULL;
	last_neuron_indices_for_each_group = NULL;

}


// Neurons Destructor
Neurons::~Neurons() {

	// Free up memory
	free(group_shapes);
	free(group_parameters);
	free(last_neuron_indices_for_each_group);

}


int Neurons::AddGroup(neuron_struct params, int group_shape[2]){
	
	int number_of_neurons_in_group = group_shape[0]*group_shape[1];
 
	if (number_of_neurons_in_group < 0) {
		printf("\nError: Population must have at least 1 neuron.\n\n");
		exit(-1);
	}

	// Update totals
	total_number_of_neurons += number_of_neurons_in_group;
	++total_number_of_groups;
	printf("total_number_of_groups: %d\n", total_number_of_groups); // Temp helper

	// Calculate new group id
	int new_group_id = total_number_of_groups - 1;

	// Add last neuron index for new group
	last_neuron_indices_for_each_group = (int*)realloc(last_neuron_indices_for_each_group,(total_number_of_groups*sizeof(int)));
	last_neuron_indices_for_each_group[new_group_id] = total_number_of_neurons;

	// Add new group shape
	group_shapes = (int**)realloc(group_shapes,(total_number_of_groups*sizeof(int*)));
	group_shapes[new_group_id] = (int*)malloc(2*sizeof(int));
	group_shapes[new_group_id] = group_shape;

	// Add new group parameters
	group_parameters = (neuron_struct*)realloc(group_parameters, (total_number_of_neurons*sizeof(neuron_struct)));
	for (int i = (total_number_of_neurons - number_of_neurons_in_group); i < total_number_of_neurons; i++){
		group_parameters[i] = params;
	}
	
	return new_group_id;
}


