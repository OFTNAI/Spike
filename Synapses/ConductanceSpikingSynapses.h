#ifndef ConductanceSPIKINGSYNAPSES_H
#define ConductanceSPIKINGSYNAPSES_H

#include "SpikingSynapses.h"
#include "../Neurons/SpikingNeurons.h"

struct conductance_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
	conductance_spiking_synapse_parameters_struct(): biological_conductance_scaling_constant_lambda(1.0) { spiking_synapse_parameters_struct(); }

	float biological_conductance_scaling_constant_lambda;
};

class ConductanceSpikingSynapses : public SpikingSynapses {

public:

	// Constructor/Destructor
	ConductanceSpikingSynapses();
	~ConductanceSpikingSynapses();

	float * synaptic_conductances_g;
	float * d_synaptic_conductances_g;

	float * recent_presynaptic_activities_C;
	float * d_recent_presynaptic_activities_C;

	float * biological_conductance_scaling_constants_lambda;
	float * d_biological_conductance_scaling_constants_lambda;

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						int delay_range[2],
						bool stdp_on,
						synapse_parameters_struct * synapse_params,
						float parameter,
						float parameter_two);

	virtual void allocate_device_pointers();
	virtual void reset_synapse_spikes();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds);
	virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
	virtual void update_presynaptic_activities(float timestep, float current_time_in_seconds);
	virtual void update_synaptic_efficacies_or_weights(float * d_recent_postsynaptic_activities_D, float timestep, float current_time_in_seconds, float * d_last_spike_time_of_each_neuron);

};

__global__ void conductance_calculate_postsynaptic_current_injection_kernal(int* d_postsynaptic_neuron_indices,
							float* d_neuron_reversal_potentials_Vhat,
							float* d_neurons_current_injections,
							size_t total_number_of_synapses,
							float * d_membrane_potentials_v,
							float * d_synaptic_conductances_g);


__global__ void conductance_update_synaptic_conductances_kernal(float timestep, 
													float * d_synaptic_conductances_g, 
													float * d_synaptic_efficacies_or_weights, 
													float * d_time_of_last_spike_to_reach_synapse,
													float * d_biological_conductance_scaling_constants_lambda,
													int total_number_of_synapses,
													float current_time_in_seconds);


__global__ void conductance_update_presynaptic_activities_C_kernal(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							bool* d_stdp,
							float timestep,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void conductance_update_synaptic_efficacies_or_weights_kernal(float * d_recent_presynaptic_activities_C,
																float * d_recent_postsynaptic_activities_D,
																float timestep,
																int* d_postsynaptic_neuron_indices,
																float* d_synaptic_efficacies_or_weights,
																float current_time_in_seconds,
																float * d_time_of_last_spike_to_reach_synapse,
																float * d_last_spike_time_of_each_neuron,
																bool* d_stdp,
																size_t total_number_of_synapses);

#endif