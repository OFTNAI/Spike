#ifndef SpikingNeurons_H
#define SpikingNeurons_H

//	CUDA library
#include <cuda.h>
#include <stdio.h>

#include "Connections.h"

#include "Neurons.h"

class SpikingNeurons : public Neurons {
public:
	// Constructor/Destructor
	SpikingNeurons();
	~SpikingNeurons();

	float * states_v;
	float * states_u;
	float * param_c;
	float * param_d;

	float * d_last_spike_time;
	float * d_states_v;
	float * d_states_u;
	float * d_param_c;
	float * d_param_d;

	// Functions
	virtual int AddGroupNew(neuron_struct *params, int shape[2]);
	virtual void initialise_device_pointersNew();
	virtual void reset_neuron_variables_and_spikesNew();

	virtual void spikingneurons_wrapper(float currtime);

};



#endif