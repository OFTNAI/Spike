// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef Connections_H
#define Connections_H
#include "Structs.h"
class Connections {
public:
	// Constructor/Destructor
	Connections();
	~Connections();
	// Variables;
	int total_number_of_connections;
	// STDP
	struct stdp_struct stdp_vars;
	void SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new);
	// Full Matrices
	int* presynaptic_neuron_indices; // Previously presyns
	int* postsynaptic_neuron_indices; // Previously postsyns
	float* weights;
	int* delays;
	int* stdp;
	// Synapse Functions
	void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						int* last_neuron_indices_for_each_neuron_group,
						int** neuron_group_shapes, 
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two);
private:
	void increment_number_of_connections(int increment);
};
// GAUSS random number generator
double randn (double mu, double sigma);
#endif