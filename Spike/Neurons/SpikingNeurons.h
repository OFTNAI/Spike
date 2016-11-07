#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cuda.h>
#include <stdio.h>

#include "../Neurons/Neurons.h"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): resting_potential_v0(-0.074f), threshold_for_action_potential_spike(0.03f), absolute_refractory_period(0.002f) { neuron_parameters_struct(); }

	float resting_potential_v0;
	float threshold_for_action_potential_spike;
	float absolute_refractory_period;
};


class SpikingNeurons : public Neurons {
public:
	// Constructor/Destructor
	SpikingNeurons();
	~SpikingNeurons();

	// Variables
	int bitarray_length;
	int bitarray_maximum_axonal_delay_in_timesteps;
	bool high_fidelity_spike_flag;
	
	// Host Pointers
	float* after_spike_reset_membrane_potentials_c;
	float* thresholds_for_action_potential_spikes;
	unsigned char* bitarray_of_neuron_spikes;

	// Device Pointers
	float* d_last_spike_time_of_each_neuron;
	float* d_membrane_potentials_v;
	float* d_thresholds_for_action_potential_spikes;
	float* d_resting_potentials;
	unsigned char * d_bitarray_of_neuron_spikes;

	// Functions
	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps,  bool high_fidelity_spike_flag);
	virtual void copy_constants_to_device();
	virtual void reset_neuron_activities();
	virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
	virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);

};


__global__ void check_for_neuron_spikes_kernel(float *d_membrane_potentials_v,
								float *d_thresholds_for_action_potential_spikes,
								float *d_resting_potentials,
								float* d_last_spike_time_of_each_neuron,
								unsigned char* d_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_neurons,
								bool high_fidelity_spike_flag);


#endif
