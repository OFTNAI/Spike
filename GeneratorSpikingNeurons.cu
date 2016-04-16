#include "GeneratorSpikingNeurons.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle
using namespace std;


// GeneratorSpikingNeurons Constructor
GeneratorSpikingNeurons::GeneratorSpikingNeurons() {

}


// GeneratorSpikingNeurons Destructor
GeneratorSpikingNeurons::~GeneratorSpikingNeurons() {

}


int GeneratorSpikingNeurons::AddGroupNew(neuron_struct * params, int group_shape[2]){
	
	int new_group_id = SpikingNeurons::AddGroupNew(params, group_shape);

	// rates = (float*)realloc(rates, (total_number_of_neurons*sizeof(float)));

	for (int i = 0; i < total_number_of_neurons; i++) {
		// rates[i] = params->rate;
	}

	return -1 * new_group_id;

}

void GeneratorSpikingNeurons::initialise_device_pointersNew() {

	SpikingNeurons::initialise_device_pointersNew();

	// CudaSafeCall(cudaMalloc((void **)&d_rates, sizeof(float)*total_number_of_neurons));
	// CudaSafeCall(cudaMalloc((void**) &d_states, total_number_of_neurons*sizeof(curandState_t)));

	GeneratorSpikingNeurons::reset_input_variables();
}


void GeneratorSpikingNeurons::reset_input_variables() {
	// CudaSafeCall(cudaMemcpy(d_rates, rates, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
	// CudaSafeCall(cudaMemcpy(d_states_v, states_v, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
}



__global__ void genupdate2(struct neuron_struct* neuronpop_variables,
							int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries);


void GeneratorSpikingNeurons::generupdate2_wrapper(int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries) {

	genupdate2<<<number_of_neuron_blocks_per_grid, threads_per_block>>> (d_neuron_variables,
												genids,
												gentimes,
												currtime,
												timestep,
												numEntries);

	CudaCheckError();
}


// Spike Generator Updating Kernel
__global__ void genupdate2(struct neuron_struct* d_neuronpop_variables,
							int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numEntries){
		// Check if the current time is one of the gen times
		if (fabs(currtime - gentimes[idx]) > 0.5*timestep) {
			// This sync seems absolutely necessary for when I spike inputs ... weird.
			d_neuronpop_variables[genids[idx]].state_u = 0.0f;
			d_neuronpop_variables[genids[idx]].state_v = -70.0f;
		} else {
			__syncthreads();
			d_neuronpop_variables[genids[idx]].state_u = 0.0f;
			d_neuronpop_variables[genids[idx]].state_v = 35.0f;
		}
	}
}
