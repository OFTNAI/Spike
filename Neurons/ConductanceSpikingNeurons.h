#ifndef ConductanceSpikingNeurons_H
#define ConductanceSpikingNeurons_H

#include <cuda.h>

#include "SpikingNeurons.h"


struct conductance_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	conductance_spiking_neuron_parameters_struct()
	// : parama(0.0f), paramb(0.0f) 
	{ 
		spiking_neuron_parameters_struct(); 
	}

	// float parama;
	// float paramb;
};


class ConductanceSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	ConductanceSpikingNeurons();
	~ConductanceSpikingNeurons();

	virtual int AddGroup(neuron_parameters_struct * group_params, int shape[2]);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();
	virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds);

	virtual void update_membrane_potentials(float timestep);

};

#endif