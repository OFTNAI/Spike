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
#include <time.h>
#include "CUDAErrorCheckHelpers.h"
#include "RecordingElectrodes.h"

#include "GeneratorSpikingNeurons.h"

// Silences the printfs
// #define QUIETSTART

__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons);



//			lastactive = vector- indicating last time synapse emitted current
void GPUDeviceComputation (
					Neurons * neurons,
					Connections * connections,
					PoissonSpikingNeurons * input_neurons,

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

	GeneratorSpikingNeurons * temp_test_generator = new GeneratorSpikingNeurons();

	
	RecordingElectrodes * recording_electrodes = new RecordingElectrodes(neurons);
	RecordingElectrodes * input_recording_electrodes = new RecordingElectrodes(input_neurons);

	neurons->initialise_device_pointers();
	connections->initialise_device_pointers();
	input_neurons->initialise_device_pointersNew();

	recording_electrodes->initialise_device_pointers();
	recording_electrodes->initialise_host_pointers();

	input_recording_electrodes->initialise_device_pointers();
	input_recording_electrodes->initialise_host_pointers();


	// THREADS&BLOCKS
	// The number of threads per block I shall keep held at 128
	int threads = 128;
	connections->set_threads_per_block_and_blocks_per_grid(threads);
	neurons->set_threads_per_block_and_blocks_per_grid(threads);
	input_neurons->set_threads_per_block_and_blocks_per_grid(threads);


	input_neurons->generate_random_states_wrapper();

	curandState_t* states;
	cudaMalloc((void**) &states, neurons->total_number_of_neurons*sizeof(curandState_t));
	// Initialise the random states
	init<<<neurons->threads_per_block, neurons->number_of_neuron_blocks_per_grid>>>(42, states, neurons->total_number_of_neurons);
	CudaCheckError();
	// Keep space for the random numbers
	float* gpu_randfloats;
	CudaSafeCall(cudaMalloc((void**) &gpu_randfloats, neurons->total_number_of_neurons*sizeof(float)));

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

				temp_test_generator->initialise_device_pointers_for_ents(numEnts, present);
				temp_test_generator->set_threads_per_block_and_blocks_per_grid(threads);
				
			}
			// Reset the variables necessary
			neurons->reset_neuron_variables_and_spikes();
			connections->reset_connection_spikes();

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
				neurons->reset_device_current_injections();
				// If there are poisson populations
				if (numPoisson > 0) {
					// First create the set of random numbers of poisson neurons
					randoms<<<neurons->threads_per_block, neurons->number_of_neuron_blocks_per_grid>>>(states, gpu_randfloats, neurons->total_number_of_neurons);
					CudaCheckError();

					// Update Poisson neuron states
					neurons->poisupdate_wrapper(gpu_randfloats, timestep);
					input_neurons->update_poisson_state_wrapper(timestep);
					
				}
				// If there are any spike generators
				if (numEnts > 0) {
					// Update those neurons corresponding to the Spike Generators
					temp_test_generator->generupdate2_wrapper(current_time_in_seconds, timestep);
				} 
				
				connections->calculate_postsynaptic_current_injection_for_connection_wrapper(neurons->d_current_injections, current_time_in_seconds);

				// Carry out LTD on appropriate synapses
				connections->ltdweights_wrapper(neurons->d_lastspiketime, current_time_in_seconds);

				// Update States of neurons
				neurons->stateupdate_wrapper(timestep);

				// Check which neurons are spiking and deal with them
				neurons->spikingneurons_wrapper(current_time_in_seconds);
				input_neurons->spikingneurons_wrapper(current_time_in_seconds);
								
				// Check which synapses to send spikes down and do it
				connections->synapsespikes_wrapper(neurons->d_lastspiketime, current_time_in_seconds);

				// // Carry out the last step, LTP!
				connections->synapseLTP_wrapper(neurons->d_lastspiketime, current_time_in_seconds);
				

				// Only save the spikes if necessary
				if (save_spikes){
					recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch, true);
					input_recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch, false);

				}
			}
			if (numEnts > 0){
				// CudaSafeCall(cudaFree(d_genids));
				// CudaSafeCall(cudaFree(d_gentimes));
			}
		}
		#ifndef QUIETSTART
		clock_t mid = clock();
		if (save_spikes) {
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC), recording_electrodes->h_total_number_of_spikes);
			printf("Number of Input Spikes: %d\n\n", input_recording_electrodes->h_total_number_of_spikes);
		
		} else {
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC));
		}
		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (save_spikes){
			recording_electrodes->write_spikes_to_file(neurons, epoch_number);
		}
	}
	
	// SIMULATION COMPLETE!
	#ifndef QUIETSTART
	// Finish the simulation and check time
	clock_t end = clock();
	float timed = float(end-begin) / CLOCKS_PER_SEC;
	printf("Simulation Complete! Time Elapsed: %f\n\n", timed);
	#endif

	recording_electrodes->save_network_state(connections);


	delete neurons;
	delete connections;
	delete recording_electrodes;

	CudaSafeCall(cudaFree(states));
	CudaSafeCall(cudaFree(gpu_randfloats));
	// CudaSafeCall(cudaFree(currentinjection));
	// Free Memory on CPU
	free(recording_electrodes->h_spikestoretimes);
	free(recording_electrodes->h_spikestoreID);

}


// Random Number Getter
__global__ void randoms(curandState_t* states, float* numbers, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		/* curand works like rand - except that it takes a state as a parameter */
		numbers[idx] = curand_uniform(&states[idx]);
	}
}

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


