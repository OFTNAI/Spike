//	CUDA code for SPIKE simulator
//
//
//	Author: Nasir Ahmad
//	Date: 9/12/2015

//  Adapted by Nasir Ahmad and James Isbister
//	Date: 23/3/2016

// For files/manipulations
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> // For random shuffle
using namespace std;


#include "CUDAcode.h"
#include "Structs.h"
#include <time.h>
#include "CUDAErrorCheckHelpers.h"
#include "RecordingElectrodes.h"
// Silences the printfs
// #define QUIETSTART


//			lastactive = vector- indicating last time synapse emitted current
void GPUDeviceComputation (
					Neurons * neurons,
					Connections * connections,

					float total_time_per_epoch,
					int number_of_epochs,
					float timestep,
					bool save_spikes,

					int numStimuli,
					int* numEntries,
					int** genids,
					float** gentimes,
					bool randomPresentation
					){


	// Creating the Device Pointers I need
	int* d_genids;
	float* d_gentimes;
	
	RecordingElectrodes * recording_electrodes = new RecordingElectrodes();

	neurons->initialise_device_pointers();
	connections->initialise_device_pointers();
	recording_electrodes->initialise_device_pointers(neurons->total_number_of_neurons);
	recording_electrodes->initialise_host_pointers(neurons->total_number_of_neurons);


	// THREADS&BLOCKS
	// The number of threads per block I shall keep held at 128
	int threads = 128;
	connections->set_threads_per_block_and_blocks_per_grid(threads);
	neurons->set_threads_per_block_and_blocks_per_grid(threads);


	dim3 threadsPerBlock(threads,1,1);
	// I now have to calculate the number of blocks ....
	int vectorblocknum = (neurons->total_number_of_neurons + threads) / threads;

	// The maximum dimension for the grid is 65535
	dim3 vectorblocksPerGrid(vectorblocknum,1,1);  
	// Temp Values which will be replaced

	int genblocknum = 1;
	dim3 genblocksPerGrid(genblocknum,1,1);


	// RANDOM NUMBERS
	// Create the random state seed trackers
	curandState_t* states;
	cudaMalloc((void**) &states, neurons->total_number_of_neurons*sizeof(curandState_t));
	// Initialise the random states
	init<<<threadsPerBlock, vectorblocksPerGrid>>>(42, states, neurons->total_number_of_neurons);
	CudaCheckError();
	// Keep space for the random numbers
	float* gpu_randfloats;
	CudaSafeCall(cudaMalloc((void**) &gpu_randfloats, neurons->total_number_of_neurons*sizeof(float)));
	// REQUIRED DATA SPACES
	float* currentinjection;
	CudaSafeCall(cudaMalloc((void**)&currentinjection, neurons->total_number_of_neurons*sizeof(float)));
	// Variables necessary
	clock_t begin = clock();

	// Poisson number
	int numPoisson = 0;
	for (int i = 0; i < neurons->total_number_of_neurons; i++){
		if (neurons->neuron_variables[i].rate != 0.0f){
			++numPoisson;
		}
	}

	// STIMULUS ORDER
	int presentorder[numStimuli];
	for (int i = 0; i < numStimuli; i++){
		presentorder[i] = i;
	}

	// SEEDING
	srand(42);

	recording_electrodes->write_initial_synaptic_weights_to_file(connections);

	// Running through all of the Epochs
	for (int epoch_number = 0; epoch_number < number_of_epochs; epoch_number++) {
		// If we want a random presentation, create the set of numbers:
		if (randomPresentation) {
			random_shuffle(&presentorder[0], &presentorder[numStimuli]);
		}
		// Running through every Stimulus
		for (int j = 0; j < numStimuli; j++){
			// Get the presentation position:
			int present = presentorder[j];
			// Get the number of entries for this specific stimulus
			size_t numEnts = numEntries[present];
			if (numEnts > 0){
				// Calculate the dimensions required:
				genblocknum = (numEnts + threads) / threads;
				// Setting up the IDs for Spike Generators;
				CudaSafeCall(cudaMalloc((void **)&d_genids, sizeof(int)*numEnts));
				CudaSafeCall(cudaMalloc((void **)&d_gentimes, sizeof(float)*numEnts));
				CudaSafeCall(cudaMemcpy(d_genids, genids[present], sizeof(int)*numEnts, cudaMemcpyHostToDevice));
				CudaSafeCall(cudaMemcpy(d_gentimes, gentimes[present], sizeof(float)*numEnts, cudaMemcpyHostToDevice));
			}
			// Reset the variables necessary
			// CAN GO INTO CLASSES EVENTUALLY
			CudaSafeCall(cudaMemcpy(neurons->d_neuron_variables, neurons->neuron_variables, sizeof(float)*neurons->total_number_of_neurons, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemset(neurons->d_lastspiketime, -1000.0f, neurons->total_number_of_neurons*sizeof(float)));
			CudaSafeCall(cudaMemset(connections->d_spikes, 0, sizeof(int)*connections->total_number_of_connections));
			CudaSafeCall(cudaMemset(connections->d_lastactive, -1000.0f, sizeof(float)*connections->total_number_of_connections));
			CudaSafeCall(cudaMemset(connections->d_spikebuffer, -1, connections->total_number_of_connections*sizeof(int)));

			// Running the Simulation!
			// Variables as Necessary
			int number_of_timesteps_per_epoch = total_time_per_epoch / timestep;
			float current_time_in_seconds = 0.0f;
			// GO!
			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_epoch; timestep_index++){
				// SIMULATION
				// Current simulation timestep
				current_time_in_seconds = float(timestep_index)*float(timestep);
				// Start by resetting all the things
				CudaSafeCall(cudaMemset(currentinjection, 0.0f, neurons->total_number_of_neurons*sizeof(float)));	
				// If there are poisson populations
				if (numPoisson > 0) {
					// First create the set of random numbers of poisson neurons
					randoms<<<vectorblocksPerGrid, threadsPerBlock>>>(states, gpu_randfloats, neurons->total_number_of_neurons);
					CudaCheckError();

					// Update Poisson neuron states
					neurons->poisupdate_wrapper(gpu_randfloats, timestep);
					
				}
				// If there are any spike generators
				if (numEnts > 0) {
					// Update those neurons corresponding to the Spike Generators
					neurons->genupdate_wrapper(d_genids,
											d_gentimes,
											current_time_in_seconds,
											timestep,
											numEnts,
											genblocknum, 
											threadsPerBlock);
					
				} 
				
				connections->calculate_postsynaptic_current_injection_for_connection_wrapper(currentinjection, current_time_in_seconds);

				// Carry out LTD on appropriate synapses
				connections->ltdweights_wrapper(neurons->d_lastspiketime, current_time_in_seconds);

				// Update States of neurons
				neurons->stateupdate_wrapper(currentinjection, timestep);

				// Check which neurons are spiking and deal with them
				neurons->spikingneurons_wrapper(current_time_in_seconds);
								
				// Check which synapses to send spikes down and do it
				connections->synapsespikes_wrapper(neurons->d_lastspiketime, current_time_in_seconds);

				// // Carry out the last step, LTP!
				connections->synapseLTP_wrapper(neurons->d_lastspiketime, current_time_in_seconds);
				

				// Only save the spikes if necessary
				if (save_spikes){
					recording_electrodes->save_spikes_to_host(neurons, current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch, neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block);
				}
			}
			if (numEnts > 0){
				CudaSafeCall(cudaFree(d_genids));
				CudaSafeCall(cudaFree(d_gentimes));
			}
		}
		#ifndef QUIETSTART
		clock_t mid = clock();
		if (save_spikes)
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC), recording_electrodes->h_total_number_of_spikes);
		else 
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC));
		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (save_spikes){
			recording_electrodes->write_spikes_to_file(neurons, epoch_number);
		}
	}
	// Finish the simulation and check time
	clock_t end = clock();
	float timed = float(end-begin) / CLOCKS_PER_SEC;
	// SIMULATION COMPLETE!
	#ifndef QUIETSTART
	printf("Simulation Complete! Time Elapsed: %f\n\n", timed);
	printf("Outputting binary files.\n");
	#endif


	// Copy back the data that we might want:
	CudaSafeCall(cudaMemcpy(connections->weights, connections->d_weights, sizeof(float)*connections->total_number_of_connections, cudaMemcpyDeviceToHost));
	// Creating and Opening all the files
	ofstream synapsepre, synapsepost, weightfile, delayfile;
	weightfile.open("results/NetworkWeights.bin", ios::out | ios::binary);
	delayfile.open("results/NetworkDelays.bin", ios::out | ios::binary);
	synapsepre.open("results/NetworkPre.bin", ios::out | ios::binary);
	synapsepost.open("results/NetworkPost.bin", ios::out | ios::binary);
	
	// Writing the data
	weightfile.write((char *)connections->weights, connections->total_number_of_connections*sizeof(float));
	delayfile.write((char *)connections->delays, connections->total_number_of_connections*sizeof(int));
	synapsepre.write((char *)connections->presynaptic_neuron_indices, connections->total_number_of_connections*sizeof(int));
	synapsepost.write((char *)connections->postsynaptic_neuron_indices, connections->total_number_of_connections*sizeof(int));

	// Close files
	weightfile.close();
	delayfile.close();
	synapsepre.close();
	synapsepost.close();


	delete neurons;
	delete connections;
	delete recording_electrodes;

	CudaSafeCall(cudaFree(states));
	CudaSafeCall(cudaFree(gpu_randfloats));
	CudaSafeCall(cudaFree(currentinjection));
	// Free Memory on CPU
	free(recording_electrodes->h_spikestoretimes);
	free(recording_electrodes->h_spikestoreID);

}



// Random Number Generator intialiser
/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&states[idx]);
	}
}

// Random Number Getter
__global__ void randoms(curandState_t* states, float* numbers, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		/* curand works like rand - except that it takes a state as a parameter */
		numbers[idx] = curand_uniform(&states[idx]);
	}
}
