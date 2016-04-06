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
//			numNeurons = Total number of neurons in simulation
//			numConnections = Total number of Synapses
//			presyns = vector- entry corresponds to pre-syn neuron ID
//			postsyns = vector- entry corresponds to post-syn neuron ID
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
//			totaltime = total time for which the sim should run
void GPUDeviceComputation (
					size_t numNeurons,
					size_t numConnections,
					int* presyns,
					int* postsyns,
					int* delays,
					float* weights,
					int* stdp,
					float* lastactive,
					struct neuron_struct* neuronpop_variables,
					int numStimuli,
					int* numEntries,
					int** genids,
					float** gentimes,
					struct stdp_struct stdp_vars,
					float timestep,
					float totaltime,
					int numEpochs,
					bool savespikes,
					bool randomPresentation
					){

	// Creating the Device Pointers I need
	int* d_presyns;
	int* d_postsyns;
	int* d_delays;
	float* d_weights;
	int* d_spikes;
	int* d_stdp;
	float* d_lastactive;
	struct neuron_struct* d_neuronpop_variables;
	float* d_lastspiketime;
	int* d_genids;
	float* d_gentimes;
	int* d_spikebuffer;
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

	// Allocate memory for data on device
	CudaSafeCall(cudaMalloc((void **)&d_presyns, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_postsyns, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_neuronpop_variables, sizeof(struct neuron_struct)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_spikebuffer, sizeof(int)*numConnections));
	CudaSafeCall(cudaMalloc((void **)&d_tempstorenum, sizeof(int)));
	CudaSafeCall(cudaMalloc((void **)&d_tempstoreID, sizeof(int)*numNeurons));
	CudaSafeCall(cudaMalloc((void **)&d_tempstoretimes, sizeof(float)*numNeurons));


	// Send the actual data!
	CudaSafeCall(cudaMemcpy(d_presyns, presyns, sizeof(int)*numConnections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsyns, postsyns, sizeof(int)*numConnections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*numConnections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, weights, sizeof(float)*numConnections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_stdp, stdp, sizeof(int)*numConnections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_neuronpop_variables, neuronpop_variables, sizeof(struct neuron_struct)*numNeurons, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_spikes, 0, sizeof(int)*numConnections));
	CudaSafeCall(cudaMemset(d_lastactive, -1000.0f, sizeof(float)*numConnections));
	CudaSafeCall(cudaMemset(d_lastspiketime, -1000.0f, numNeurons*sizeof(float)));
	CudaSafeCall(cudaMemset(d_spikebuffer, -1, numConnections*sizeof(int)));
	CudaSafeCall(cudaMemset(d_tempstorenum, 0, sizeof(int)));
	CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*numNeurons));
	CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*numNeurons));
	h_tempspikenum = (int*)malloc(sizeof(int));
	h_tempstoreID = (int*)malloc(sizeof(int)*numNeurons);
	h_tempstoretimes = (float*)malloc(sizeof(float)*numNeurons);
	h_tempspikenum[0] = 0;



	// THREADS&BLOCKS
	// The number of threads per block I shall keep held at 128
	int threads = 128;
	dim3 threadsPerBlock(threads,1,1);
	// I now have to calculate the number of blocks ....
	int connblocknum = (numConnections + threads) / threads;
	int vectorblocknum = (numNeurons + threads) / threads;
	// The maximum dimension for the grid is 65535
	dim3 connblocksPerGrid(connblocknum,1,1);
	dim3 vectorblocksPerGrid(vectorblocknum,1,1);
	// Temp Values which will be replaced
	int genblocknum = 1;
	dim3 genblocksPerGrid(genblocknum,1,1);


	// RANDOM NUMBERS
	// Create the random state seed trackers
	curandState_t* states;
	cudaMalloc((void**) &states, numNeurons*sizeof(curandState_t));
	// Initialise the random states
	init<<<threadsPerBlock, vectorblocksPerGrid>>>(42, states, numNeurons);
	CudaCheckError();
	// Keep space for the random numbers
	float* gpu_randfloats;
	CudaSafeCall(cudaMalloc((void**) &gpu_randfloats, numNeurons*sizeof(float)));
	// REQUIRED DATA SPACES
	float* currentinjection;
	CudaSafeCall(cudaMalloc((void**)&currentinjection, numNeurons*sizeof(float)));
	// Variables necessary
	clock_t begin = clock();

	// Poisson number
	int numPoisson = 0;
	for (int i = 0; i < numNeurons; i++){
		if (neuronpop_variables[i].rate != 0.0f){
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
	initweightfile.write((char *)weights, numConnections*sizeof(float));
	initweightfile.close();

	// Running through all of the Epochs
	for (int i = 0; i < numEpochs; i++) {
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
			CudaSafeCall(cudaMemcpy(d_neuronpop_variables, neuronpop_variables, sizeof(float)*numNeurons, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemset(d_spikes, 0, sizeof(int)*numConnections));
			CudaSafeCall(cudaMemset(d_lastactive, -1000.0f, sizeof(float)*numConnections));
			CudaSafeCall(cudaMemset(d_lastspiketime, -1000.0f, numNeurons*sizeof(float)));
			CudaSafeCall(cudaMemset(d_spikebuffer, -1, numConnections*sizeof(int)));

			// Running the Simulation!
			// Variables as Necessary
			int numtimesteps = totaltime / timestep;
			float currtime = 0.0f;
			// GO!
			for (int k = 0; k < numtimesteps; k++){
				// SIMULATION
				// Current simulation timestep
				currtime = (float(k))*(float(timestep));
				// Start by resetting all the things
				CudaSafeCall(cudaMemset(currentinjection, 0.0f, numNeurons*sizeof(float)));	
				// If there are poisson populations
				if (numPoisson > 0) {
					// First create the set of random numbers of poisson neurons
					randoms<<<vectorblocksPerGrid, threadsPerBlock>>>(states, gpu_randfloats, numNeurons);
					CudaCheckError();
					// Update Poisson neuron states
					poisupdate<<<vectorblocksPerGrid, threadsPerBlock>>>(gpu_randfloats,
																		d_neuronpop_variables,
																		timestep,
																		numNeurons);
					CudaCheckError();
				}
				// If there are any spike generators
				if (numEnts > 0) {
					// Update those neurons corresponding to the Spike Generators
					genupdate<<<genblocknum, threadsPerBlock>>> (d_neuronpop_variables,
																	d_genids,
																	d_gentimes,
																	currtime,
																	timestep,
																	numEnts);
					CudaCheckError();
				} 
				// Calculate current injections to cells
				currentcalc<<<connblocksPerGrid, threadsPerBlock>>>(d_spikes,
																	d_weights,
																	d_lastactive,
																	d_postsyns,
																	currentinjection,
																	currtime,
																	numConnections,
																	numNeurons);
				CudaCheckError();
				// Carry out LTD on appropriate synapses
				ltdweights<<<connblocksPerGrid, threadsPerBlock>>>(d_lastactive,
																	d_weights,
																	d_stdp,
																	d_lastspiketime,
																	d_postsyns,
																	currtime,
																	stdp_vars,
																	numConnections,
																	numNeurons);
				CudaCheckError();
				// Update States of neurons
				stateupdate<<<vectorblocksPerGrid, threadsPerBlock>>>(d_neuronpop_variables,
																	currentinjection,
																	timestep,
																	numNeurons);
				CudaCheckError();
				// Check which neurons are spiking and deal with them
				spikingneurons<<<vectorblocksPerGrid, threadsPerBlock>>>(d_neuronpop_variables,
																		d_lastspiketime,
																		currtime,
																		numNeurons);
				CudaCheckError();
				// Check which synapses to send spikes down and do it
				synapsespikes<<<connblocksPerGrid, threadsPerBlock>>>(d_presyns,
																		d_delays,
																		d_spikes,
																		d_lastspiketime,
																		d_spikebuffer,
																		currtime,
																		numConnections,
																		numNeurons);
				CudaCheckError();
				// Carry out the last step, LTP!
				synapseLTP<<<connblocksPerGrid, threadsPerBlock>>>(d_postsyns,
																	d_lastspiketime,
																	d_stdp,
																	d_lastactive,
																	d_weights,
																	stdp_vars,
																	currtime,
																	numConnections,
																	numNeurons);
				CudaCheckError();

				// Only save the spikes if necessary
				if (savespikes){
					// Storing the spikes that have occurred in this timestep
					spikeCollect<<<vectorblocksPerGrid, threadsPerBlock>>>(d_lastspiketime,
																		d_tempstorenum,
																		d_tempstoreID,
																		d_tempstoretimes,
																		currtime,
																		numNeurons);
					CudaCheckError();
					if (((k % 1) == 0) || (k == (numtimesteps-1))){
						// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
						CudaSafeCall(cudaMemcpy(&h_tempspikenum[0], &d_tempstorenum[0], (sizeof(int)), cudaMemcpyDeviceToHost));
						// Ensure that we don't have too many
						if (h_tempspikenum[0] > numNeurons){
							// ERROR!
							printf("Spike recorder has been overloaded! Reduce threshold. Exiting ...\n");
							exit(-1);
						}
						// Deal with them!
						if ((h_tempspikenum[0] >= (0.25*numNeurons)) ||  (k == (numtimesteps - 1))){
							// Allocate some memory for them:
							h_spikestoreID = (int*)realloc(h_spikestoreID, sizeof(int)*(h_spikenum + h_tempspikenum[0]));
							h_spikestoretimes = (float*)realloc(h_spikestoretimes, sizeof(float)*(h_spikenum + h_tempspikenum[0]));
							// Copy the data from device to host
							CudaSafeCall(cudaMemcpy(h_tempstoreID, 
													d_tempstoreID, 
													(sizeof(int)*(numNeurons)), 
													cudaMemcpyDeviceToHost));
							CudaSafeCall(cudaMemcpy(h_tempstoretimes, 
													d_tempstoretimes, 
													sizeof(float)*(numNeurons), 
													cudaMemcpyDeviceToHost));
							// Pop all of the times where they need to be:
							for (int l = 0; l < h_tempspikenum[0]; l++){
								h_spikestoreID[h_spikenum + l] = h_tempstoreID[l];
								h_spikestoretimes[h_spikenum + l] = h_tempstoretimes[l];
							}
							// Reset the number on the device
							CudaSafeCall(cudaMemset(&d_tempstorenum[0], 0, sizeof(int)));
							CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*numNeurons));
							CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*numNeurons));
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
		if (savespikes)
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", i, (float(mid-begin) / CLOCKS_PER_SEC), h_spikenum);
		else 
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", i, (float(mid-begin) / CLOCKS_PER_SEC));
		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (savespikes){
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
			CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*numNeurons));
			CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*numNeurons));
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
	CudaSafeCall(cudaMemcpy(weights, d_weights, sizeof(float)*numConnections, cudaMemcpyDeviceToHost));
	// Creating and Opening all the files
	ofstream synapsepre, synapsepost, weightfile, delayfile;
	weightfile.open("results/NetworkWeights.bin", ios::out | ios::binary);
	delayfile.open("results/NetworkDelays.bin", ios::out | ios::binary);
	synapsepre.open("results/NetworkPre.bin", ios::out | ios::binary);
	synapsepost.open("results/NetworkPost.bin", ios::out | ios::binary);
	
	// Writing the data
	weightfile.write((char *)weights, numConnections*sizeof(float));
	delayfile.write((char *)delays, numConnections*sizeof(int));
	synapsepre.write((char *)presyns, numConnections*sizeof(int));
	synapsepost.write((char *)postsyns, numConnections*sizeof(int));

	// Close files
	weightfile.close();
	delayfile.close();
	synapsepre.close();
	synapsepost.close();


	// Free Memory on GPU
	CudaSafeCall(cudaFree(d_presyns));
	CudaSafeCall(cudaFree(d_postsyns));
	CudaSafeCall(cudaFree(d_delays));
	CudaSafeCall(cudaFree(d_weights));
	CudaSafeCall(cudaFree(d_spikes));
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_neuronpop_variables));
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
__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeurons) {
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&states[idx]);
	}
}

// Random Number Getter
__global__ void randoms(curandState_t* states, float* numbers, size_t numNeurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeurons) {
		/* curand works like rand - except that it takes a state as a parameter */
		numbers[idx] = curand_uniform(&states[idx]);
	}
}

// Current Calculation Kernel
__global__ void currentcalc(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsyns,
							float* currentinj,
							float currtime,
							size_t numConns,
							size_t numNeurons){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (numConns)) {
		// Decrememnt Spikes
		d_spikes[idx] -= 1;
		if (d_spikes[idx] == 0) {
			// Get locations of weights and lastactive
			atomicAdd(&currentinj[d_postsyns[idx]], d_weights[idx]);
			// Change lastactive
			d_lastactive[idx] = currtime;
			// Done!
		}
	}
	__syncthreads();
}

// Synapses carrying spikes
__global__ void synapsespikes(int* d_presyns,
								int* d_delays,
								int* d_spikes,
								float* d_lastspiketime,
								int* d_spikebuffer,
								float currtime,
								size_t numConns,
								size_t numNeurons){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (numConns)) {
		// Reduce the spikebuffer by 1
		d_spikebuffer[idx] -= 1;
		// Check if the neuron PRE has just fired and if the synapse exists
		if (d_lastspiketime[d_presyns[idx]] == currtime){
			// Update the spikes with the correct delay
			if (d_spikes[idx] <= 0){
				d_spikes[idx] = d_delays[idx];
			} else if (d_spikebuffer[idx] <= 0){
				d_spikebuffer[idx] = d_delays[idx];
			}
		}
		// If there is no waiting spike
		if (d_spikes[idx] <= 0) {
			// Use the buffer if necessary
			if (d_spikebuffer[idx] > 0) {
				d_spikes[idx] = d_spikebuffer[idx];
			} else {
				d_spikes[idx] = -1;
				d_spikebuffer[idx] = -1;
			}
		}
		// If the buffer has a smaller time than the spike, switch them
		if ((d_spikebuffer[idx] > 0) && (d_spikebuffer[idx] < d_spikes[idx])){
			int temp = d_spikes[idx];
			d_spikes[idx] = d_spikebuffer[idx];
			d_spikebuffer[idx] = temp;
		}
	}
}

// Collect Spikes
__global__ void spikeCollect(float* d_lastspiketime,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float currtime,
								size_t numNeurons){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeurons) {
		// If a neuron has fired
		if (d_lastspiketime[idx] == currtime) {
			// Increase the number of spikes stored
			int i = atomicAdd(&d_tempstorenum[0], 1);
			// In the location, add the id and the time
			d_tempstoreID[i] = idx;
			d_tempstoretimes[i] = currtime;
		}
	}
	__syncthreads();
}
