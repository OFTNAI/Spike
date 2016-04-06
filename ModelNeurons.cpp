//	ModelNeurons Class C++
//	ModelNeurons.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015
//
//  Adapted from NeuronPopulations by Nasir Ahmad and James Isbister
//	Date: 6/4/2016

#include "ModelNeurons.h"
#include <stdlib.h>
#include <stdio.h>

// ModelNeurons Constructor
ModelNeurons::ModelNeurons() {

	total_number_of_neurons = 0;
	total_number_of_groups = 0;

	group_shapes = NULL;
	group_parameters = NULL;
	last_neuron_indices_for_each_group = NULL;

}

// ModelNeurons Destructor
ModelNeurons::~ModelNeurons() {

	// Just need to free up the memory
	free(group_shapes);
	free(group_parameters);
	free(last_neuron_indices_for_each_group);

}

// Add Group Function
//	INPUT:
//		Izhikevich parameter list {a, b, c, d}
//		Group Shape
int ModelNeurons::AddGroup(neuron_struct params, int group_shape[2]){
	// Check that it is within bounds

	int number_of_neurons_in_group = group_shape[0]*group_shape[1];
 
	if (number_of_neurons_in_group < 0) {
		printf("\nError: Population must have at least 1 neuron.\n\n");
		exit(-1);
	}

	// Update the totals
	total_number_of_neurons += number_of_neurons_in_group;
	++total_number_of_groups;
	printf("total_number_of_groups: %d\n", total_number_of_groups);

	int new_group_id = total_number_of_groups - 1;

	last_neuron_indices_for_each_group = (int*)realloc(last_neuron_indices_for_each_group,(total_number_of_groups*sizeof(int)));
	last_neuron_indices_for_each_group[new_group_id] = total_number_of_neurons;


	// Allocate space for the new neurons
	group_shapes = (int**)realloc(group_shapes,(total_number_of_groups*sizeof(int*)));
	group_shapes[new_group_id] = (int*)malloc(2*sizeof(int));

	group_parameters = (neuron_struct*)realloc(group_parameters, (total_number_of_neurons*sizeof(neuron_struct)));

	// Fill the new entries in the pointers
	group_shapes[new_group_id] = group_shape;
	for (int i = (total_number_of_neurons - number_of_neurons_in_group); i < total_number_of_neurons; i++){

		group_parameters[i] = params;
	
	}
	
	return new_group_id;
}


