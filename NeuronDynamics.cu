//	Neuron Dynamics CUDA Code
//	NeuronDynamics.cu
//
//	Author: Nasir Ahmad
//	Date: 15/03/2016

#include "NeuronDynamics.h"
#include <stdlib.h>
#include <stdio.h>

// // Poisson Updating Kernel
// __global__ void poisupdate(float* d_randoms, 
// 							struct neuron_struct* d_neuronpop_variables,
// 							float timestep,
// 							size_t numNeurons){
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (idx < numNeurons){
// 		// if the randomnumber is LT the rate
// 		if (d_randoms[idx] < (d_neuronpop_variables[idx].rate*timestep)){
// 			d_neuronpop_variables[idx].state_u = 0.0f;
// 			d_neuronpop_variables[idx].state_v = 35.0f;
// 		} else if (d_neuronpop_variables[idx].rate != 0.0f) {
// 			d_neuronpop_variables[idx].state_u = 0.0f;
// 			d_neuronpop_variables[idx].state_v = -70.0f;
// 		}
// 	}
// 	__syncthreads();
// }

// // Spike Generator Updating Kernel
// __global__ void genupdate(struct neuron_struct* d_neuronpop_variables,
// 							int* genids,
// 							float* gentimes,
// 							float currtime,
// 							float timestep,
// 							size_t numEntries){
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (idx < numEntries){
// 		// Check if the current time is one of the gen times
// 		if (fabs(currtime - gentimes[idx]) > 0.5*timestep) {
// 			// This sync seems absolutely necessary for when I spike inputs ... weird.
// 			d_neuronpop_variables[genids[idx]].state_u = 0.0f;
// 			d_neuronpop_variables[genids[idx]].state_v = -70.0f;
// 		} else {
// 			__syncthreads();
// 			d_neuronpop_variables[genids[idx]].state_u = 0.0f;
// 			d_neuronpop_variables[genids[idx]].state_v = 35.0f;
// 		}
// 	}
// }

// // // State Update
// // __global__ void stateupdate(struct neuron_struct* d_neuronpop_variables,
// // 							float* currentinj,
// // 							float timestep,
// // 							size_t numNeurons){
// // 	// We require the equation timestep in ms:
// // 	float eqtimestep = timestep*1000.0f;
// // 	// Get thread IDs
// // 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
// // 	if (idx < numNeurons) {
// // 		// Update the neuron states according to the Izhikevich equations
// // 		float v_update = 0.04f*d_neuronpop_variables[idx].state_v*d_neuronpop_variables[idx].state_v + 
// // 							5.0f*d_neuronpop_variables[idx].state_v + 140 - d_neuronpop_variables[idx].state_u + currentinj[idx];
// // 		d_neuronpop_variables[idx].state_v += eqtimestep*v_update;
// // 		d_neuronpop_variables[idx].state_u += eqtimestep*(d_neuronpop_variables[idx].parama * (d_neuronpop_variables[idx].paramb*d_neuronpop_variables[idx].state_v - 
// // 							d_neuronpop_variables[idx].state_u));
// // 	}
// // 	__syncthreads();
// // }

// // Spiking Neurons
// __global__ void spikingneurons(struct neuron_struct* d_neuronpop_variables,
// 								float* d_lastspiketime,
// 								float currtime,
// 								size_t numNeurons){
// 	// Get thread IDs
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (idx < numNeurons) {
// 		// First checking if neuron has spiked:
// 		if (d_neuronpop_variables[idx].state_v >= 30.0f){
// 			// Reset the values of these neurons
// 			d_neuronpop_variables[idx].state_v = d_neuronpop_variables[idx].paramc;
// 			d_neuronpop_variables[idx].state_u += d_neuronpop_variables[idx].paramd;
// 			// Update the last spike times of these neurons
// 			d_lastspiketime[idx] = currtime;
// 		}
// 	}
// 	__syncthreads();
// }