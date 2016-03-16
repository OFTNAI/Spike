// STDP Dynamics CUDA Header
// STDPDynamics.h
//
//	Author: Nasir Ahmad
//	Date: 16/03/2015

#ifndef STDPDynamics_H
#define STDPDynamics_H

#include "Structs.h"

__global__ void ltdweights(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t numConns,
							size_t numNeurons);
__global__ void synapseLTP(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns,
							size_t numNeurons);
#endif