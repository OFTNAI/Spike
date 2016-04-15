// Structures Header
// Structs.h
//
//	Author: Nasir Ahmad
//	Date: 15/03/2016

#ifndef Structs_H
#define Structs_H


struct neuron_struct_2 {};

struct izhikevich_neuron_struct_2 : neuron_struct_2 {
	izhikevich_neuron_struct_2(): parama(0.0f), paramb(0.0f), paramc(0.0f), paramd(0.0f), state_v(-70.0f), state_u(0.0f) { }   // default Constructor
	float parama;
	float paramb;
	float paramc;
	float paramd;
	// State variables
	float state_v;
	float state_u;
};

struct input_neuron_struct_2 : neuron_struct_2 {
	input_neuron_struct_2(): state_v(-70.0f), state_u(0.0f), rate(0.0f) { }   // default Constructor
	float state_v;
	float state_u;
	// Rate for poisson
	float rate;
};



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

// struct neuron_struct_test : neuron_struct {
// 	neuron_struct_test(): test(0.0f) { neuron_struct(); }   // default Constructor

// 	float test;
// };


struct input_struct {
	// State variables
	float state_v;
	float state_u;
	// Rate for poisson
	float rate;
};

// STDP Parameters
struct stdp_struct {
	stdp_struct(): w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;
};

#endif