//	ModelNeurons Class Header
//	ModelNeurons.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015
//
//  Adapted from NeuronPopulations by Nasir Ahmad and James Isbister
//	Date: 6/4/2016

#ifndef ModelNeurons_H
#define ModelNeurons_H

#include "Structs.h"

class ModelNeurons{
public:
	// Constructor/Destructor
	ModelNeurons();
	~ModelNeurons();

	// Totals
	int total_number_of_neurons;
	int total_number_of_groups;

	// Group parameters, shapes and indices
	neuron_struct *group_parameters;
	int **group_shapes;
	int *last_neuron_indices_for_each_group;
	
	// Functions
	int AddGroup(neuron_struct params, int shape[2]);

};
#endif