// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef Connections_H
#define Connections_H

#include "Neurons.h"


// STDP Parameters
struct stdp_struct {
	stdp_struct(): w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;
};


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

	// Device pointers
	int* d_presynaptic_neuron_indices;
	int* d_postsynaptic_neuron_indices;
	int* d_delays;
	float* d_weights;
	int* d_spikes;
	int* d_stdp;
	float* d_lastactive;
	int* d_spikebuffer;

	// Synapse Functions
	void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two);

	void initialise_device_pointers();
	void reset_connection_spikes();
	void set_threads_per_block_and_blocks_per_grid(int threads);

	void calculate_postsynaptic_current_injection_for_connection_wrapper(float* d_neurons_current_injections, float current_time_in_seconds);
	void synapsespikes_wrapper(float* d_neurons_last_spike_time, float* d_input_neurons_last_spike_time, float current_time_in_seconds);
	void ltdweights_wrapper(float* d_lastspiketime, float current_time_in_seconds);
	void synapseLTP_wrapper(float* d_lastspiketime, float current_time_in_seconds);


private:
	dim3 number_of_connection_blocks_per_grid;
	dim3 threads_per_block;

	void increment_number_of_connections(int increment);
};
// GAUSS random number generator
double randn (double mu, double sigma);
#endif