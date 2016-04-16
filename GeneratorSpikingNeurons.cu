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

void GeneratorSpikingNeurons::initialise_device_pointers_for_ents(int numEntsParam, int present) {

	SpikingNeurons::initialise_device_pointersNew();

	numEnts = numEntsParam;

	CudaSafeCall(cudaMalloc((void **)&d_genids, sizeof(int)*numEnts));
	CudaSafeCall(cudaMalloc((void **)&d_gentimes, sizeof(float)*numEnts));

	GeneratorSpikingNeurons::reset_input_variables(present);
}


void GeneratorSpikingNeurons::reset_input_variables(int present) {
	CudaSafeCall(cudaMemcpy(d_genids, genids[present], sizeof(int)*numEnts, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_gentimes, gentimes[present], sizeof(float)*numEnts, cudaMemcpyHostToDevice));
}


void GeneratorSpikingNeurons::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	SpikingNeurons::set_threads_per_block_and_blocks_per_grid(threads);

	int genblocknum = (numEnts + threads) / threads;
	genblocksPerGrid.x = genblocknum;
}



__global__ void genupdate2(float *d_states_v,
							float *d_states_u,
							int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries);


void GeneratorSpikingNeurons::generupdate2_wrapper(float currtime,
							float timestep) {

	genupdate2<<<genblocksPerGrid, threads_per_block>>> (d_states_v,
														d_states_u,
														d_genids,
														d_gentimes,
														currtime,
														timestep,
														numEnts);

	CudaCheckError();
}


// Spike Generator Updating Kernel
__global__ void genupdate2(float *d_states_v,
							float *d_states_u,
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
			d_states_u[idx] = 0.0f;
			d_states_v[idx] = -70.0f;
		} else {
			__syncthreads();
			d_states_u[idx] = 0.0f;
			d_states_v[idx] = 35.0f;
		}
	}
}
