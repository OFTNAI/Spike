#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cuda.h>
#include <stdio.h>

#include "../Neurons/Neurons.h"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): resting_potential_v0(-0.074f), threshold_for_action_potential_spike(0.03f) { neuron_parameters_struct(); }

	float resting_potential_v0;
	float threshold_for_action_potential_spike;
};


class SpikingNeurons : public Neurons {
public:
	// Constructor/Destructor
	SpikingNeurons();
	~SpikingNeurons();

	//Group-wise parameters
	float * after_spike_reset_membrane_potentials_c;
	float * thresholds_for_action_potential_spikes;

	//Changing device variables
	float * d_last_spike_time_of_each_neuron;
	float * d_membrane_potentials_v;

	//Device group-wise parameters
	float * d_thresholds_for_action_potential_spikes;
	float * d_resting_potentials;

	//LIF extra
	float * recent_postsynaptic_activities_D;
	float * d_recent_postsynaptic_activities_D;


	// Functions
	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();

	virtual void update_membrane_potentials(float timestep);
	virtual void check_for_neuron_spikes(float current_time_in_seconds);
	virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds);

};


__global__ void check_for_neuron_spikes_kernal(float *d_membrane_potentials_v,
								float *d_thresholds_for_action_potential_spikes,
								float *d_resting_potentials,
								float* d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								size_t total_number_of_neurons);


#endif