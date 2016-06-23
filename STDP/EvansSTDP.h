// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

#include "../Neurons/Neurons.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>


// STDP Parameters
// Temporarily Synapse member (should move to SpikingNeurons)
struct stdp_struct {
	stdp_struct(): w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;
};



class STDP {

public:

	// Constructor/Destructor
	STDP();
	~STDP();

	// STDP
	// Temporarily Synapse members (should move to SpikingNeurons)
	struct stdp_struct stdp_vars;
	void SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new);

};

#endif