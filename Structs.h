// Structures Header
// Structs.h
//
//	Author: Nasir Ahmad
//	Date: 15/03/2016

#ifndef Structs_H
#define Structs_H

// Izhikevich Neuron Parameters
struct neuron_struct {
	neuron_struct(): parama(0.0f), paramb(0.0f), paramc(0.0f), paramd(0.0f), state_v(-70.0f), state_u(0.0f), rate(0.0f) { }   // default Constructor
	float parama;
	float paramb;
	float paramc;
	float paramd;
	// State variables
	float state_v;
	float state_u;
	// Rate for poisson
	float rate;
};

#endif