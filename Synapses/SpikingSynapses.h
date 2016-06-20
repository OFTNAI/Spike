#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.h"
#include "../Neurons/SpikingNeurons.h"


struct spiking_synapse_parameters_struct : synapse_parameters_struct {
	spiking_synapse_parameters_struct(): stdp_on(true) { synapse_parameters_struct(); }

	bool stdp_on;
	float delay_range[2];

};

class SpikingSynapses : public Synapses {

public:

	// Constructor/Destructor
	SpikingSynapses();
	~SpikingSynapses();

	// Full Matrices
	int* delays;
	bool* stdp;

	// Device pointers
	int* d_delays;
	bool* d_stdp;
	
	int* d_spikes_travelling_to_synapse;
	int* d_spikes_travelling_to_synapse_buffer;
	float* d_time_of_last_spike_to_reach_synapse;
	

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params,
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
	virtual void update_presynaptic_activities(float timestep, float current_time_in_seconds);
	virtual void update_synaptic_efficacies_or_weights(float * d_recent_postsynaptic_activities_D, float current_time_in_seconds, float * d_last_spike_time_of_each_neuron);

	virtual void move_spikes_towards_synapses(float* d_last_spike_time_of_each_neuron, float* d_input_neurons_last_spike_time, float current_time_in_seconds);
	virtual void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	virtual void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);


};

__global__ void check_for_synapse_spike_arrival_kernal(int* d_spikes_travelling_to_synapse,
							float* d_time_of_last_spike_to_reach_synapse,
							float current_time_in_seconds,
							size_t total_number_of_synapses);


__global__ void move_spikes_towards_synapses_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes_travelling_to_synapse,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								int* d_spikes_travelling_to_synapse_buffer,
								float currtime,
								size_t total_number_of_synapses,
								float* d_time_of_last_spike_to_reach_synapse);

#endif