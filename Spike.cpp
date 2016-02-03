//	Spike Simulator AIO
//
//
//	Author: Nasir Ahmad
//	Date: 8/12/2015

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "Spike.h"


// Constructor
Spike::Spike(){
	// Poisson Populations
	numPoisson = 0;
	poisson = NULL;
	poissonrate = NULL;
	poissonmask = NULL;
	// Spike Generators
	numStimuli = 0;
	numEntries = NULL;
	genids = NULL;
	gentimes = NULL;
	// Things for checking
	numPops = 0;
	numConnects = 0;
	// Default parameters
	timestep = 0.001f;
	a_plus = 0.005f;
	a_minus = -0.015f;
	tau_plus = 0.015f;
	tau_minus = 0.025f;
	w_max = 60.0f;

	#ifndef QUIETSTART
		// Say Hi to the user:
		printf("\nWelcome to the SPIKE.\n\n");
		printf("Setting up Populations and Synapses: ");
		fflush(stdout);
	#endif
}

// Destructor
Spike::~Spike(){
	free(poisson);
	free(poissonrate);
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
int Spike::CreateNeurons(int number, char type[], float params[]){
	// Check which type of population it is
	if (strcmp(type, "izh") == 0){
		// If we have an izhikevich population
		int ID = population.AddPopulation(number, params);
		return ID;
	} else if (strcmp(type, "poisson") == 0){
		// If we have a poisson population
		// Add space to store indices
		poisson = (int**)realloc(poisson, sizeof(int*)*(numPoisson+1));
		poisson[numPoisson] = (int*)malloc(2*sizeof(int));
		// Set those spaces equal to the initial and final values of number
		poisson[numPoisson][0] = population.numNeurons;
		poisson[numPoisson][1] = population.numNeurons + number;
		// Set the rate as required
		poissonrate = (float*)realloc(poissonrate, sizeof(float)*(numPoisson+1));
		poissonrate[numPoisson] = params[0];
		// Have the neural population created with arbitrary parameters
		float izhparams[] = {0.0f, 0.0f, 0.0f, 0.0f};
		// Create the population
		int ID = population.AddPopulation(number, izhparams);
		++numPoisson;
		return ID;
	} else if (strcmp(type, "gen") == 0){
		// Create the neural population
		// Have the neural population created with arbitrary parameters
		float izhparams[] = {0.0f, 0.0f, 0.0f, 0.0f};
		int ID = population.AddPopulation(number, izhparams);
		return ID;
	} else {
		// Not recognised the population
		printf("Unrecognised Population Type");
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
void Spike::CreateConnection(int pre, 
							int post, 
							char style[], 
							float weights[2], 
							float delays[2],
							bool stdp,
							float parameter){
	// Convert delays from time to # timesteps
	int stepdelays[2] = {int(round(delays[0]/timestep)), int(round(delays[1]/timestep))};
	// Ensure that the value of delays is sufficiently large.
	if ((stepdelays[0] < 1) || (stepdelays[1] < 1)) {
		printf("Delay range be at least one timestep. Exiting ...\n");
		exit(-1);
	}
	// Create the connection!
	synconnects.AddConnection(pre, 
							post, 
							population.numperPop, 
							style, 
							weights,
							stepdelays,
							stdp,
							parameter);
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

// Poisson Mask Creation
// No input required
void Spike::PoissonMask(){
	// Allocate the memory
	poissonmask = (float *)malloc(population.numNeurons*sizeof(float));
	// Set initial value for all neurons to zero
	for (int i = 0; i < population.numNeurons; i++){
		poissonmask[i] = 0.0f;
	}
	// Go through the poisson neurons and set the rates
	for (int i = 0; i < numPoisson; i++) {
		// Get the indices of the poisson populations
		int startid = poisson[i][0];
		int endid = poisson[i][1];
		// Set the mask equal to the rate for each poisson neuron
		for (int x = startid; x < endid; x++){
			poissonmask[x] = poissonrate[i];
		}
	}
}

// Command for running the simulation
// No input required
void Spike::Run(float totaltime, int numEpochs, bool saveSpikes, bool randompresentation){
	// Get the Poisson Mask Sorted
	PoissonMask();

	#ifndef QUIETSTART
	// Give the user some feedback
	printf("\n\n----------------------------------\n");
	printf("Simulation Beginning\n");
	printf("Time Step: %f\nNumber of Stimuli: %d\nNumber of Epochs: %d\n\n", timestep, numStimuli, numEpochs);
	printf("Total Number of Neurons: %d\n", population.numNeurons);
	printf("Total Number of Synapses: %d\n\n", synconnects.numconnections);
	if (randompresentation)
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
					population.state_v,
					population.state_u,
					population.parama,
					population.paramb,
					population.paramc,
					population.paramd,
					poissonmask,
					numPoisson,
					numStimuli,
					numEntries,
					genids,
					gentimes,
					synconnects.w_max,
					synconnects.a_minus,
					synconnects.a_plus,
					synconnects.tau_minus,
					synconnects.tau_plus,
					timestep,
					totaltime,
					numEpochs,
					saveSpikes,
					randompresentation);
}



