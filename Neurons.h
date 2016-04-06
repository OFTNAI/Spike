//	Neurons Class Header
//	Neurons.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015
//
//  Adapted from NeuronPopulations by Nasir Ahmad and James Isbister
//	Date: 6/4/2016

#ifndef Neurons_H
#define Neurons_H

#include "Structs.h"

class Neurons{
public:
	// Constructor/Destructor
	Neurons();
	~Neurons();

	// Totals
	int total_number_of_neurons;
	int total_number_of_groups;

	// Group parameters, shapes and indices
	neuron_struct *group_parameters; //Currently actually neuron wise. Should eventually change
	int **group_shapes;
	int *last_neuron_indices_for_each_group;
	
	// Functions
	int AddGroup(neuron_struct params, int shape[2]);

};
#endif