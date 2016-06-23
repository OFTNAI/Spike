// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.h"
#include "../Neurons/SpikingNeurons.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>


// STDP Parameters
struct stdp_parameters_struct {
	stdp_parameters_struct() {}
};


class STDP {
public:

	// Constructor/Destructor
	STDP();
	~STDP();

	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, stdp_parameters_struct* stdp_parameters);
	// Initialize STDP
	virtual void Initialize_STDP();
	// STDP
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
	// Reset
	virtual void Reset_STDP();
};

#endif