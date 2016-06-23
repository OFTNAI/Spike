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

// Setting personal STDP parameters
void STDP::SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new){
	// Set the values
	stdp_vars.w_max = w_max_new;
	stdp_vars.a_minus = a_minus_new;
	stdp_vars.a_plus = a_plus_new;
	stdp_vars.tau_minus = tau_minus_new;
	stdp_vars.tau_plus = tau_plus_new;
}