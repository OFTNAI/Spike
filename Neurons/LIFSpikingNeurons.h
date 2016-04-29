#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

#include <cuda.h>

#include "SpikingNeurons.h"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct()
	// : parama(0.0f), paramb(0.0f) 
	{ 
		spiking_neuron_parameters_struct(); 
	}

	// float parama;
	// float paramb;
};


class LIFSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	LIFSpikingNeurons();
	~LIFSpikingNeurons();

	// float * param_a;
	// float * param_b;

	// float * d_param_a;
	// float * d_param_b;

	virtual int AddGroup(neuron_parameters_struct * group_params, int shape[2]);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();

	virtual void update_membrane_potentials(float timestep);

};

#endif