// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

// Get Synapses Class
#include "../Synapses/Synapses.h"

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
}


class STDP {
public:

	// Constructor/Destructor
	STDP(Synapses syns);
	~STDP();

	// Synapses to use for STDP
	Synapses network_synapses;
	// STDP
	virtual void ImplementSTDPRule(stdp_parameters_struct * stdp_params);
};

#endif