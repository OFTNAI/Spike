#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cuda.h>
#include <stdio.h>

#include "../Neurons/Neurons.h"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): after_spike_reset_membrane_potential_c(-70.0f), paramd(0.0f) { neuron_parameters_struct(); }

	float after_spike_reset_membrane_potential_c;
	float paramd;
};


class SpikingNeurons : public Neurons {
public:
	// Constructor/Destructor
	SpikingNeurons();
	~SpikingNeurons();

	float * d_last_spike_times;

	float * after_spike_reset_membrane_potentials_c;
	float * param_d;

	float * d_membrane_potentials_v;
	float * d_states_u;
	float * d_after_spike_reset_membrane_potentials_c;
	float * d_param_d;

	// Functions
	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointers();
	virtual void reset_neurons();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	virtual void update_neuron_states(float timestep);
	virtual void check_for_neuron_spikes(float currtime);

};



#endif