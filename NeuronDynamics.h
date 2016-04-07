// Neuron Dynamics CUDA Header
// NeuronDynamics.h
//
//	Author: Nasir Ahmad
//	Date: 15/03/2015

#ifndef NeuronDynamics_H
#define NeuronDynamics_H

#include "Structs.h"

// __global__ void poisupdate(float* d_randoms, 
// 							struct neuron_struct* neuronpop_variables,
// 							float timestep,
// 							size_t numNeurons);
// __global__ void genupdate(struct neuron_struct* neuronpop_variables,
// 							int* genids,
// 							float* gentimes,
// 							float currtime,
// 							float timestep,
// 							size_t numEntries);
// __global__ void stateupdate(struct neuron_struct* neuronpop_variables,
// 							float* currentinj,
// 							float timestep,
// 							size_t numNeurons);
// __global__ void spikingneurons(struct neuron_struct* neuronpop_variables,
// 								float* d_lastspiketime,
// 								float currtime,
// 								size_t numNeurons);
#endif