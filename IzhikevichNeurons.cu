#include "IzhikevichNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"


// IzhikevichNeurons Constructor
IzhikevichNeurons::IzhikevichNeurons() {

	izhikevich_neuron_variables = NULL;

}


// IzhikevichNeurons Destructor
IzhikevichNeurons::~IzhikevichNeurons() {

}


int IzhikevichNeurons::AddGroupNew(neuron_struct *params, int group_shape[2]){

	int new_group_id = Neurons::AddGroupNew(params, group_shape);

	izhikevich_neuron_variables = (izhikevich_neuron_struct*)realloc(izhikevich_neuron_variables, (total_number_of_neurons*sizeof(izhikevich_neuron_struct)));
	for (int i = (total_number_of_neurons - number_of_neurons_in_new_group); i < total_number_of_neurons; i++){
		izhikevich_neuron_variables[i] = *((izhikevich_neuron_struct*)params);
	}

	return new_group_id;
}


void IzhikevichNeurons::initialise_device_pointersNew() {
 	
 	Neurons::initialise_device_pointersNew();

	CudaSafeCall(cudaMalloc((void **)&d_izhikevich_neuron_variables, sizeof(struct izhikevich_neuron_struct)*total_number_of_neurons));
	
	reset_neuron_variables_and_spikesNew();
}

void IzhikevichNeurons::reset_neuron_variables_and_spikesNew() {

	Neurons::reset_neuron_variables_and_spikesNew();	

	CudaSafeCall(cudaMemcpy(d_izhikevich_neuron_variables, izhikevich_neuron_variables, sizeof(struct izhikevich_neuron_struct)*total_number_of_neurons, cudaMemcpyHostToDevice));

}



// // CUDA __global__ function declarations
// // NOTE: these are NOT MEMBER FUNCTIONS
// // They are called by their corresponding wrapper member function

// __global__ void poisupdate(float* d_randoms, 
// 							struct neuron_struct* d_neuronpop_variables,
// 							float timestep,
// 							size_t numNeurons);

// __global__ void genupdate(struct neuron_struct* neuronpop_variables,
// 							int* genids,
// 							float* gentimes,
// 							float currtime,
// 							float timestep,
// 							size_t numEntries);

// __global__ void spikingneurons(struct neuron_struct* neuronpop_variables,
// 							float* d_lastspiketime,
// 							float currtime,
// 							size_t numNeurons);

// __global__ void stateupdate(struct neuron_struct* neuronpop_variables,
// 							float* currentinj,
// 							float timestep,
// 							size_t numNeurons);



// // Wrapper member function definitions
// // See NOTE above
// void IzhikevichNeurons::poisupdate_wrapper(float* d_randoms, float timestep) {

// 	poisupdate<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_randoms,
// 														d_neuron_variables,
// 														timestep,
// 														total_number_of_neurons);
// 	CudaCheckError();
// }


// void IzhikevichNeurons::genupdate_wrapper(int* genids,
// 							float* gentimes,
// 							float currtime,
// 							float timestep,
// 							size_t numEntries,
// 							int genblocknum, 
// 							dim3 threadsPerBlock) {

// 	genupdate<<<genblocknum, threadsPerBlock>>> (d_neuron_variables,
// 												genids,
// 												gentimes,
// 												currtime,
// 												timestep,
// 												numEntries);

// 	CudaCheckError();
// }


// void IzhikevichNeurons::spikingneurons_wrapper(float currtime) {

// 	spikingneurons<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_neuron_variables,
// 																		d_lastspiketime,
// 																		currtime,
// 																		total_number_of_neurons);

// 	CudaCheckError();
// }


// void IzhikevichNeurons::stateupdate_wrapper(float* current_injection,
// 							float timestep) {

// 	stateupdate<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(d_neuron_variables,
// 																	current_injection,
// 																	timestep,
// 																	total_number_of_neurons);

// 	CudaCheckError();
// }




// // CUDA __global__ function definitions
// // These are called by the Neurons class member functions
// // May have to vary names if 'including' more than one subclass

// // Poisson Updating Kernal
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


// // State Update
// __global__ void stateupdate(struct neuron_struct* d_neuronpop_variables,
// 							float* currentinj,
// 							float timestep,
// 							size_t numNeurons){
// 	// We require the equation timestep in ms:
// 	float eqtimestep = timestep*1000.0f;
// 	// Get thread IDs
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (idx < numNeurons) {
// 		// Update the neuron states according to the Izhikevich equations
// 		float v_update = 0.04f*d_neuronpop_variables[idx].state_v*d_neuronpop_variables[idx].state_v + 
// 							5.0f*d_neuronpop_variables[idx].state_v + 140 - d_neuronpop_variables[idx].state_u + currentinj[idx];
// 		d_neuronpop_variables[idx].state_v += eqtimestep*v_update;
// 		d_neuronpop_variables[idx].state_u += eqtimestep*(d_neuronpop_variables[idx].parama * (d_neuronpop_variables[idx].paramb*d_neuronpop_variables[idx].state_v - 
// 							d_neuronpop_variables[idx].state_u));
// 	}
// 	__syncthreads();
// }

