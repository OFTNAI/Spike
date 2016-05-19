#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cuda.h>
#include <stdio.h>

#include "../Neurons/Neurons.h"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): resting_potential_v0(-0.074f), threshold_for_action_potential_spike(0.03f), paramd(6.0f), reversal_potential_Vhat(0.0f) { neuron_parameters_struct(); }

	float resting_potential_v0;
	float threshold_for_action_potential_spike;
	float paramd;
	float reversal_potential_Vhat;
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

	//Izhikevich extra
	float * d_states_u;
	float * param_d;
	float * d_param_d;

	//LIF extra
	float * recent_postsynaptic_activities_D;
	float * d_recent_postsynaptic_activities_D;
	float * reversal_potentials_Vhat;
	float * d_reversal_potentials_Vhat;


	// Functions
	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	virtual void update_membrane_potentials(float timestep);
	virtual void check_for_neuron_spikes(float currtime);
	virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds);

};



#endif