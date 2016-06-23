//	STDP Class C++
//	STDP.cpp
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "STDP.h"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// STDP Constructor
STDP::STDP() {
}

// STDP Destructor
STDP::~STDP() {
}

void STDP::Set_STDP_Parameters(SpikingSynapses* synapses, stdp_parameters_struct* stdp_parameters){
}

// Setting personal STDP parameters
void STDP::Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds){
}