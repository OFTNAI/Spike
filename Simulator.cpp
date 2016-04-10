// 	Simulator Class
// 	Simulator.cpp
//
//	Original Author: Nasir Ahmad
//	Date: 8/12/2015
//	Originally Spike.cpp
//  
//  Adapted by Nasir Ahmad and James Isbister
//	Date: 23/3/2016

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "Simulator.h"
#include "Constants.h"


// Constructor
Simulator::Simulator(){
	// Spike Generators
	numStimuli = 0;
	numEntries = NULL;
	genids = NULL;
	gentimes = NULL;
	// Default parameters
	timestep = 0.001f;

	connections = new Connections();
	
	#ifndef QUIETSTART
		// Say Hi to the user:
		printf("\nWelcome to the SPIKE.\n\n");
		printf("Setting up Populations and Synapses: \n\n");
		fflush(stdout);
	#endif
}


// Destructor
Simulator::~Simulator(){
	free(numEntries);
	free(genids);
	free(gentimes);
}


// Timestep Setting function
void Simulator::SetTimestep(float timest){

	printf("timest = %d", connections->total_number_of_connections);
	if (connections->total_number_of_connections == 0){
		timestep = timest;
	} else {
		printf("You must set the timestep before creating any synapses. Exiting ...\n\n");
		exit(-1);
	}
}


void Simulator::SetNeuronType(Neurons * neurons_parameter) {
	neurons = neurons_parameter;
}

void Simulator::SetInputType(Inputs * inputs_parameter) {
	inputs = inputs_parameter;
}


int Simulator::AddNeuronGroup(neuron_struct group_params, int group_shape[2]) {
	if (neurons == NULL) {
		printf("Please call SetNeuronType before adding neuron groups. Exiting ...\n\n");
		exit(-1);
	}
	int neuron_group_id = neurons->AddGroup(group_params, group_shape);
	return neuron_group_id;
}

int Simulator::AddInputGroup(input_struct group_params, int group_shape[2]) {
	if (inputs == NULL) {
		printf("Please call SetInputType before adding inputs groups. Exiting ...\n\n");
		exit(-1);
	}
	int input_group_id = inputs->AddGroup(group_params, group_shape);
	return input_group_id;
}



void Simulator::AddConnectionGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							int connectivity_type,
							float weight_range[2], 
							float delay_range[2],
							bool stdp_on,
							float parameter,
							float parameter_two) {
	
	// Convert delay range from time to number of timesteps
	int delay_range_in_timesteps[2] = {int(round(delay_range[0]/timestep)), int(round(delay_range[1]/timestep))};
	if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
		printf("\nDelay range must be at least one timestep. Exiting ...\n\n");
		exit(-1);
	}

	connections->AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons->last_neuron_indices_for_each_group,
							neurons->group_shapes,
							connectivity_type, 
							weight_range,
							delay_range_in_timesteps,
							stdp_on,
							parameter,
							parameter_two);
}


void Simulator::Run(float total_time_per_epoch, int number_of_epochs, bool save_spikes, bool random_presentation){
	#ifndef QUIETSTART
	// Give the user some feedback
	printf("\n\n----------------------------------\n");
	printf("Simulation Beginning\n");
	printf("Time Step: %f\nNumber of Stimuli: %d\nNumber of Epochs: %d\n\n", timestep, numStimuli, number_of_epochs);
	printf("Total Number of Neurons: %d\n", neurons->total_number_of_neurons);
	printf("Total Number of Synapses: %d\n\n", connections->total_number_of_connections);
	if (random_presentation)
		printf("Stimuli to be presented in a random order.\n");
	if (save_spikes)
		printf("Spikes shall be saved.\n");
	printf("----------------------------------\n\nBeginning ...\n\n");
	#endif

	// Check how many stimuli their are and do something about it:
	if (numStimuli == 0){
		++numStimuli;
		numEntries = (int*)realloc(numEntries, sizeof(int)*numStimuli);
		numEntries[0] = 0;
	}
	// Ensure that there is at least one epoch
	if (number_of_epochs == 0){
		printf("Error. There must be at least one epoch. Exiting ...\n\n");
		exit(-1);
	}

	// Do the SPIKING SIMULATION!
	GPUDeviceComputation (
					neurons,
					connections,

					// Could put following 4 in Simulator parameters dictionary
					total_time_per_epoch,
					number_of_epochs,
					timestep,
					save_spikes,

					numStimuli,
					numEntries,
					genids,
					gentimes,
					random_presentation);


}


// Spike Generator Spike Creation
// INPUT:
//		Population ID
//		Stimulus ID
//		Number of Neurons
//		Number of entries in our arrays
//		Array of generator indices (neuron IDs)
//		Corresponding array of the spike times for each instance
void Simulator::CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes){
	// We have to ensure that we have created space for the current stimulus.
	if ((numStimuli - 1) < stimulusid) {
		// Check what the difference is and quit if it is too high
		if ((stimulusid - (numStimuli - 1)) > 1){
			// Error Quit
			printf("Error: Stimuli not created in order. Exiting ...\n");
			exit(-1);
		}
		// If it isn't greater than 1, make space!
		++numStimuli;
		numEntries = (int*)realloc(numEntries, sizeof(int)*numStimuli);
		genids = (int**)realloc(genids, sizeof(int*)*numStimuli);
		gentimes = (float**)realloc(gentimes, sizeof(float*)*numStimuli);
		// Initialize stuff
		genids[stimulusid] = NULL;
		gentimes[stimulusid] = NULL;
		numEntries[stimulusid] = 0;
	}
	// Spike generator populations are necessary
	// Create space for the new ids
	
	genids[stimulusid] = (int*)realloc(genids[stimulusid], 
								sizeof(int)*(spikenumber + numEntries[stimulusid]));
	gentimes[stimulusid] = (float*)realloc(gentimes[stimulusid], 
								sizeof(float)*(spikenumber + numEntries[stimulusid]));
	
	// Check where the neuron population starts
	int startnum = 0;
	if (popID > 0) {
		startnum = neurons->last_neuron_indices_for_each_group[popID-1];
	}
	
	// Assign the genid values according to how many neurons exist already
	for (int i = 0; i < spikenumber; i++){
		genids[stimulusid][numEntries[stimulusid]+i] = ids[i] + startnum;
		gentimes[stimulusid][numEntries[stimulusid]+i] = spiketimes[i];
	}
	// Increment the number of entries the generator population
	numEntries[stimulusid] += spikenumber;
	
}



// // Synapse weight loading
// // INPUT:
// //		Number of weights that you are inputting
// //		The array in which the weights are located
// void Simulator::LoadWeights(int numWeights,
// 						float* newWeights){
// 	// Check if you have the correct number of weights
// 	if (numWeights != synconnects.numconnections){
// 		// Error if not
// 		printf("The number of weights being loaded is not equivalent to the model. Exiting \n");
// 		exit(-1);
// 	}
// 	// Continuing and applying the weights
// 	for (int i=0; i < numWeights; i++){
// 		synconnects.weights[i] = newWeights[i];
// 	}
// }