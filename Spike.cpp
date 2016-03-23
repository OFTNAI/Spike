//	Spike Simulator AIO
//
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "Spike.h"
#include "Constants.h"


// Constructor
Spike::Spike(){
	// Spike Generators
	numStimuli = 0;
	numEntries = NULL;
	genids = NULL;
	gentimes = NULL;
	// Default parameters
	timestep = 0.001f;
	numPops = 0;
	numConnects = 0;
	
	#ifndef QUIETSTART
		// Say Hi to the user:
		printf("\nWelcome to the SPIKE.\n\n");
		printf("Setting up Populations and Synapses: ");
		fflush(stdout);
	#endif
}

// Destructor
Spike::~Spike(){
	free(numEntries);
	free(genids);
	free(gentimes);
}

// Timestep Setting function
void Spike::SetTimestep(float timest){
	if (synconnects.numconnections == 0){
		timestep = timest;
	} else {
		printf("You must set the timestep before creating any synapses. Exiting ...\n");
		exit(-1);
	}
}

// Neural Population Creation
//	INPUT:
//		Number of Neurons
//		Type of Neural Population e.g. "izh"
//		Izhikevich Parameter List {a, b, c, d}
int Spike::CreateNeurons(int number, int neuron_type, struct neuron_struct params, int shape[2]){
	if (shape[0]*shape[1] != number){
		printf("Shape of the neuron population and number in population do not match. Exiting. \n");
		exit(-1);
	}
	// Check which type of population it is
	if (neuron_type == NEURON_TYPE_IZHIKEVICH){
		// If we have an izhikevich population
		int ID = population.AddPopulation(number, params, shape);
		return ID;
	} else if (neuron_type == NEURON_TYPE_POISSON){
		// If we have a poisson population
		// Create the population. The params struct should contain the rate.
		int ID = population.AddPopulation(number, params, shape);
		return ID;
	} else if (neuron_type == NEURON_TYPE_GEN){
		// Create the neural population
		// Have the neural population created with arbitrary parameters
		int ID = population.AddPopulation(number, params, shape);
		return ID;
	} else {
		// Not recognised the population
		printf("Unrecognised Population Type\n");
		exit(-1);
	}
}


// Spike Generator Spike Creation
// INPUT:
//		Population ID
//		Stimulus ID
//		Number of Neurons
//		Number of entries in our arrays
//		Array of generator indices (neuron IDs)
//		Corresponding array of the spike times for each instance
void Spike::CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes){
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
		startnum = population.numperPop[popID-1];
	}
	
	// Assign the genid values according to how many neurons exist already
	for (int i = 0; i < spikenumber; i++){
		genids[stimulusid][numEntries[stimulusid]+i] = ids[i] + startnum;
		gentimes[stimulusid][numEntries[stimulusid]+i] = spiketimes[i];
	}
	// Increment the number of entries the generator population
	numEntries[stimulusid] += spikenumber;
	
}

// Synapse Connection Creation
// INPUT:
//		Pre-synaptic population ID
//		Post-synaptic population ID
//		Style of connection (e.g. "all_to_all")
//		Weight Range
//		Delay Range
//		STDP True/False
void Spike::CreateConnection(int prepop, 
							int postpop, 
							int connectivity_type,
							float weights[2], 
							float delays[2],
							bool stdp,
							float parameter,
							float parameter_two){
	// Convert delays from time to # timesteps
	int stepdelays[2] = {int(round(delays[0]/timestep)), int(round(delays[1]/timestep))};
	// Ensure that the value of delays is sufficiently large.
	if ((stepdelays[0] < 1) || (stepdelays[1] < 1)) {
		printf("Delay range be at least one timestep. Exiting ...\n");
		exit(-1);
	}
	// Create the connection!
	synconnects.AddConnection(prepop, 
							postpop, 
							population.numperPop,
							population.neuronpop_shapes,
							connectivity_type, 
							weights,
							stepdelays,
							stdp,
							parameter,
							parameter_two);
}

// Synapse weight loading
// INPUT:
//		Number of weights that you are inputting
//		The array in which the weights are located
void Spike::LoadWeights(int numWeights,
						float* newWeights){
	// Check if you have the correct number of weights
	if (numWeights != synconnects.numconnections){
		// Error if not
		printf("The number of weights being loaded is not equivalent to the model. Exiting \n");
		exit(-1);
	}
	// Continuing and applying the weights
	for (int i=0; i < numWeights; i++){
		synconnects.weights[i] = newWeights[i];
	}
}

// Command for running the simulation
// No input required
void Spike::Run(float totaltime, int numEpochs, bool saveSpikes, bool randomPresentation){
	#ifndef QUIETSTART
	// Give the user some feedback
	printf("\n\n----------------------------------\n");
	printf("Simulation Beginning\n");
	printf("Time Step: %f\nNumber of Stimuli: %d\nNumber of Epochs: %d\n\n", timestep, numStimuli, numEpochs);
	printf("Total Number of Neurons: %d\n", population.numNeurons);
	printf("Total Number of Synapses: %d\n\n", synconnects.numconnections);
	if (randomPresentation)
		printf("Stimuli to be presented in a random order.\n");
	if (saveSpikes)
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
	if (numEpochs == 0){
		printf("Error. There must be at least one epoch. Exiting ...\n\n");
		exit(-1);
	}

	// Do the SPIKING SIMULATION!
	GPUDeviceComputation (
					population.numNeurons,
					synconnects.numconnections,
					synconnects.presyns,
					synconnects.postsyns,
					synconnects.delays,
					synconnects.weights,
					synconnects.stdp,
					synconnects.lastactive,
					population.neuronpop_variables,
					numStimuli,
					numEntries,
					genids,
					gentimes,
					synconnects.stdp_vars,
					timestep,
					totaltime,
					numEpochs,
					saveSpikes,
					randomPresentation);
}



