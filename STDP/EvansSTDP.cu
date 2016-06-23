//	Evans STDP Class C++
//	EvansSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "EvansSTDP.h"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// STDP Constructor
EvansSTDP::EvansSTDP() {

}

// STDP Destructor
EvansSTDP::~EvansSTDP() {

}

// Implementation of the STDP Rule for Irina's Model
void EvansSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, stdp_parameters_struct* stdp_parameters){
	stdp_params = (evans_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
}

// Run the STDP
void EvansSTDP::Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds){
	// Update
}
