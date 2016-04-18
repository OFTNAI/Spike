#ifndef IzhikevichSpikingNeurons_H
#define IzhikevichSpikingNeurons_H

//	CUDA library
#include <cuda.h>

#include "SpikingNeurons.h"

struct izhikevich_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	izhikevich_spiking_neuron_parameters_struct(): parama(0.0f), paramb(0.0f) { spiking_neuron_parameters_struct(); }

	float parama;
	float paramb;
};


struct izhikevich_neuron_struct : public neuron_struct {
	izhikevich_neuron_struct(): test(0.0f) { neuron_struct(); }   // default Constructor

	float test;
};

class IzhikevichSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	IzhikevichSpikingNeurons();
	~IzhikevichSpikingNeurons();

	float * param_a;
	float * param_b;

	float * d_param_a;
	float * d_param_b;

	virtual int AddGroupNew(neuron_parameters_struct * group_params, int shape[2]);
	virtual void initialise_device_pointersNew();
	virtual void reset_neuron_variables_and_spikesNew();

	void izhikevich_state_update_wrapper(float timestep);

};

#endif