//	STDP Class C++
//	STDP.cpp
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "STDP.h"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


// STDP Constructor
STDP::STDP(Synapses syns) {
	network_synapses = syns;
}

// STDP Destructor
STDP::~STDP() {

}

// Setting personal STDP parameters
void STDP::ImplementSTDPRule(stdp_parameters_struct * stdp_params){
}