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
// Silences the printfs
//#define QUIETSTART


// CUDA Spiking Simulator Run
//	INPUT (incomplete):
//			total_number_of_neurons = Total number of neurons in simulation
//			total_number_of_connections = Total number of Synapses
//			presynaptic_neuron_indices = vector- entry corresponds to pre-syn neuron ID
//			postsynaptic_neuron_indices = vector- entry corresponds to post-syn neuron ID
//			delays = vector- entry corresponds to axonal delay of syn
//			weight = vector- entry correponds to synaptic weight
//			spikes = vector- entry corresponds to countdown of spike on syn
//			stdp = vector- entry corresponding to whether or not syn is stdp
//			lastactive = vector- indicating last time synapse emitted current
//			parameters=
//			w_max = stdp parameter
//			a_minus = stdp parameter
//			a_plus = stdp parameter
//			tau_minus = stdp parameter
//			tau_plus = stdp parameter
//			timestep = timestep of the simulation
//			total_time_per_epoch = total time for which the sim should run per epoch
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


	// Useful to have local versions of totals as used a lot
	size_t total_number_of_neurons = neurons->total_number_of_neurons;
	size_t total_number_of_connections = connections->total_number_of_connections;


	// Creating the Device Pointers I need
	int* d_genids;
	float* d_gentimes;
	
	// I need a location in which to store spikes
	int* d_tempstorenum;
	int* d_tempstoreID;
	float* d_tempstoretimes;
	// And some on host for copy back
	int* h_tempstoreID;
	float* h_tempstoretimes;
	// This is the location in which they shall occasionally be put
	int* h_tempspikenum;
	int h_spikenum = 0;
	int* h_spikestoreID = NULL;
	float* h_spikestoretimes = NULL;


	// For saving spikes (Make seperate class)
	CudaSafeCall(cudaMalloc((void **)&d_tempstoreID, sizeof(int)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_tempstoretimes, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_tempstorenum, sizeof(int)));

	// Send data to device: data for saving spikes
	CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_tempstorenum, 0, sizeof(int)));


	neurons->initialise_device_pointers();
	connections->initialise_device_pointers();


	// Allocate host data
	h_tempspikenum = (int*)malloc(sizeof(int));
	h_tempstoreID = (int*)malloc(sizeof(int)*total_number_of_neurons);
	h_tempstoretimes = (float*)malloc(sizeof(float)*total_number_of_neurons);
	h_tempspikenum[0] = 0;



	// THREADS&BLOCKS
	// The number of threads per block I shall keep held at 128
	int threads = 128;
	connections->set_threads_per_block_and_blocks_per_grid(threads);
	neurons->set_threads_per_block_and_blocks_per_grid(threads);


	dim3 threadsPerBlock(threads,1,1);
	// I now have to calculate the number of blocks ....
	int vectorblocknum = (total_number_of_neurons + threads) / threads;

	// The maximum dimension for the grid is 65535
	dim3 vectorblocksPerGrid(vectorblocknum,1,1);  
	// Temp Values which will be replaced

	int genblocknum = 1;
	dim3 genblocksPerGrid(genblocknum,1,1);


	// RANDOM NUMBERS
	// Create the random state seed trackers
	curandState_t* states;
	cudaMalloc((void**) &states, total_number_of_neurons*sizeof(curandState_t));
	// Initialise the random states
	init<<<threadsPerBlock, vectorblocksPerGrid>>>(42, states, total_number_of_neurons);
	CudaCheckError();
	// Keep space for the random numbers
	float* gpu_randfloats;
	CudaSafeCall(cudaMalloc((void**) &gpu_randfloats, total_number_of_neurons*sizeof(float)));
	// REQUIRED DATA SPACES
	float* currentinjection;
	CudaSafeCall(cudaMalloc((void**)&currentinjection, total_number_of_neurons*sizeof(float)));
	// Variables necessary
	clock_t begin = clock();

	// Poisson number
	int numPoisson = 0;
	for (int i = 0; i < total_number_of_neurons; i++){
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

	// INITIAL WEIGHT OUTPUT
	ofstream initweightfile;
	initweightfile.open("results/NetworkWeights_Initial.bin", ios::out | ios::binary);
	initweightfile.write((char *)connections->weights, total_number_of_connections*sizeof(float));
	initweightfile.close();

	// Running through all of the Epochs
	for (int i = 0; i < number_of_epochs; i++) {
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
			CudaSafeCall(cudaMemcpy(neurons->d_neuron_variables, neurons->neuron_variables, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemset(neurons->d_lastspiketime, -1000.0f, total_number_of_neurons*sizeof(float)));
			CudaSafeCall(cudaMemset(connections->d_spikes, 0, sizeof(int)*total_number_of_connections));
			CudaSafeCall(cudaMemset(connections->d_lastactive, -1000.0f, sizeof(float)*total_number_of_connections));
			CudaSafeCall(cudaMemset(connections->d_spikebuffer, -1, total_number_of_connections*sizeof(int)));

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
				CudaSafeCall(cudaMemset(currentinjection, 0.0f, total_number_of_neurons*sizeof(float)));	
				// If there are poisson populations
				if (numPoisson > 0) {
					// First create the set of random numbers of poisson neurons
					randoms<<<vectorblocksPerGrid, threadsPerBlock>>>(states, gpu_randfloats, total_number_of_neurons);
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
					// Storing the spikes that have occurred in this timestep
					spikeCollect<<<vectorblocksPerGrid, threadsPerBlock>>>(neurons->d_lastspiketime,
																		d_tempstorenum,
																		d_tempstoreID,
																		d_tempstoretimes,
																		current_time_in_seconds,
																		total_number_of_neurons);
					CudaCheckError();

					if (((timestep_index % 1) == 0) || (timestep_index == (number_of_timesteps_per_epoch-1))){
						// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
						CudaSafeCall(cudaMemcpy(&h_tempspikenum[0], &d_tempstorenum[0], (sizeof(int)), cudaMemcpyDeviceToHost));
						// Ensure that we don't have too many
						if (h_tempspikenum[0] > total_number_of_neurons){
							// ERROR!
							printf("Spike recorder has been overloaded! Reduce threshold. Exiting ...\n");
							exit(-1);
						}
						// Deal with them!
						if ((h_tempspikenum[0] >= (0.25*total_number_of_neurons)) ||  (timestep_index == (number_of_timesteps_per_epoch - 1))){
							// Allocate some memory for them:
							h_spikestoreID = (int*)realloc(h_spikestoreID, sizeof(int)*(h_spikenum + h_tempspikenum[0]));
							h_spikestoretimes = (float*)realloc(h_spikestoretimes, sizeof(float)*(h_spikenum + h_tempspikenum[0]));
							// Copy the data from device to host
							CudaSafeCall(cudaMemcpy(h_tempstoreID, 
													d_tempstoreID, 
													(sizeof(int)*(total_number_of_neurons)), 
													cudaMemcpyDeviceToHost));
							CudaSafeCall(cudaMemcpy(h_tempstoretimes, 
													d_tempstoretimes, 
													sizeof(float)*(total_number_of_neurons), 
													cudaMemcpyDeviceToHost));
							// Pop all of the times where they need to be:
							for (int l = 0; l < h_tempspikenum[0]; l++){
								h_spikestoreID[h_spikenum + l] = h_tempstoreID[l];
								h_spikestoretimes[h_spikenum + l] = h_tempstoretimes[l];
							}
							// Reset the number on the device
							CudaSafeCall(cudaMemset(&d_tempstorenum[0], 0, sizeof(int)));
							CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*total_number_of_neurons));
							CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*total_number_of_neurons));
							// Increase the number on host
							h_spikenum += h_tempspikenum[0];
							h_tempspikenum[0] = 0;
						}
					}
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
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", i, (float(mid-begin) / CLOCKS_PER_SEC), h_spikenum);
		else 
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", i, (float(mid-begin) / CLOCKS_PER_SEC));
		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (save_spikes){
			// Get the names
			string file = "results/Epoch" + to_string(i) + "_";
			// Open the files
			ofstream spikeidfile, spiketimesfile;
			spikeidfile.open((file + "SpikeIDs.bin"), ios::out | ios::binary);
			spiketimesfile.open((file + "SpikeTimes.bin"), ios::out | ios::binary);
			// Send the data
			spikeidfile.write((char *)h_spikestoreID, h_spikenum*sizeof(int));
			spiketimesfile.write((char *)h_spikestoretimes, h_spikenum*sizeof(float));
			// Close the files
			spikeidfile.close();
			spiketimesfile.close();

			// Reset the spike store
			// Host values
			h_spikenum = 0;
			h_tempspikenum[0] = 0;
			// Free/Clear Device stuff
			// Reset the number on the device
			CudaSafeCall(cudaMemset(&d_tempstorenum[0], 0, sizeof(int)));
			CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*total_number_of_neurons));
			CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*total_number_of_neurons));
			// Free malloced host stuff
			free(h_spikestoreID);
			free(h_spikestoretimes);
			h_spikestoreID = NULL;
			h_spikestoretimes = NULL;
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
	CudaSafeCall(cudaMemcpy(connections->weights, connections->d_weights, sizeof(float)*total_number_of_connections, cudaMemcpyDeviceToHost));
	// Creating and Opening all the files
	ofstream synapsepre, synapsepost, weightfile, delayfile;
	weightfile.open("results/NetworkWeights.bin", ios::out | ios::binary);
	delayfile.open("results/NetworkDelays.bin", ios::out | ios::binary);
	synapsepre.open("results/NetworkPre.bin", ios::out | ios::binary);
	synapsepost.open("results/NetworkPost.bin", ios::out | ios::binary);
	
	// Writing the data
	weightfile.write((char *)connections->weights, total_number_of_connections*sizeof(float));
	delayfile.write((char *)connections->delays, total_number_of_connections*sizeof(int));
	synapsepre.write((char *)connections->presynaptic_neuron_indices, total_number_of_connections*sizeof(int));
	synapsepost.write((char *)connections->postsynaptic_neuron_indices, total_number_of_connections*sizeof(int));

	// Close files
	weightfile.close();
	delayfile.close();
	synapsepre.close();
	synapsepost.close();


	delete neurons;
	delete connections;

	CudaSafeCall(cudaFree(states));
	CudaSafeCall(cudaFree(gpu_randfloats));
	CudaSafeCall(cudaFree(currentinjection));
	// Free Memory on CPU
	free(h_spikestoretimes);
	free(h_spikestoreID);

}





//////////////////////////////
/////// CUDA FUNCTIONS ///////
//////////////////////////////


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


// Collect Spikes
__global__ void spikeCollect(float* d_lastspiketime,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float current_time_in_seconds,
								size_t total_number_of_neurons){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		// If a neuron has fired
		if (d_lastspiketime[idx] == current_time_in_seconds) {
			// Increase the number of spikes stored
			int i = atomicAdd(&d_tempstorenum[0], 1);
			// In the location, add the id and the time
			d_tempstoreID[i] = idx;
			d_tempstoretimes[i] = current_time_in_seconds;
		}
	}
	__syncthreads();
}
