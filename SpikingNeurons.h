#ifndef SpikingNeurons_H
#define SpikingNeurons_H

//	CUDA library
#include <cuda.h>
#include <stdio.h>

#include "Connections.h"

#include "Neurons.h"

struct spiking_neuron_parameters_struct {
	spiking_neuron_parameters_struct(): state_v(-70.0f), state_u(0.0f), paramc(0.0f), paramd(0.0f) { neuron_parameters_struct(); }

	float state_v;
	float state_u;
	float paramc;
	float paramd;
};



class SpikingNeurons : public Neurons {
public:
	// Constructor/Destructor
	SpikingNeurons();
	~SpikingNeurons();

	float * states_v;
	float * states_u;
	float * param_c;
	float * param_d;

	float * d_states_v;
	float * d_states_u;
	float * d_param_c;
	float * d_param_d;

	// Functions
	virtual int AddGroupNew(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointersNew();
	virtual void reset_neuron_variables_and_spikesNew();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	virtual void spikingneurons_wrapper(float currtime);

};



#endif