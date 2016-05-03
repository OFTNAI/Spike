#ifndef LIFSPIKINGSYNAPSES_H
#define LIFSPIKINGSYNAPSES_H

#include "SpikingSynapses.h"
#include "../Neurons/SpikingNeurons.h"


class LIFSpikingSynapses : public SpikingSynapses {

public:

	// Constructor/Destructor
	LIFSpikingSynapses();
	~LIFSpikingSynapses();

	float * synaptic_conductances_g;
	float * d_synaptic_conductances_g;

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
	virtual void sort_synapses_by_postsynaptic_neuron_indices();

	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds);
	virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
	virtual void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	virtual void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);

};

#endif