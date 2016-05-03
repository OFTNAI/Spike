#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.h"
#include "../Neurons/SpikingNeurons.h"


class SpikingSynapses : public Synapses {

public:

	// Constructor/Destructor
	SpikingSynapses();
	~SpikingSynapses();

	// Full Matrices
	int* delays;
	int* stdp;

	// Device pointers
	int* d_delays;
	int* d_stdp;
	
	int* d_spikes_travelling_to_synapse;
	int* d_spikes_travelling_to_synapse_buffer;
	float* d_time_of_last_postsynaptic_activation_for_each_synapse;
	

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two);

	virtual void allocate_device_pointers();
	virtual void reset_synapse_spikes();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	virtual void check_for_synapse_spike_arrival(float current_time_in_seconds);
	virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds);

	virtual void move_spikes_towards_synapses(float* d_last_spike_time_of_each_neuron, float* d_input_neurons_last_spike_time, float current_time_in_seconds);
	virtual void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	virtual void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);


};

#endif