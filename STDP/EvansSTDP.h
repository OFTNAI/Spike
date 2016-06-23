// Evans STDP Class Header
// EvansSTDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef Evans_STDP_H
#define Evans_STDP_H

// Get Synapses Class
#include "../Synapses/Synapses.h"
#include "../STDP/STDP.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>


// STDP Parameters
struct evans_stdp_parameters_struct : stdp_parameters_struct {
	evans_stdp_parameters_struct() : w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;

};


class EvansSTDP : public STDP{
public:

	// Constructor/Destructor
	EvansSTDP();
	~EvansSTDP();

	struct Evans_stdp_parameters_struct* stdp_params;
	SpikingSynapses* syns;

	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, stdp_parameters_struct* stdp_parameters);
	// STDP
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	// Updates for this model
};

#endif