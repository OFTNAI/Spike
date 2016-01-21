//	Synapse Class C++
//	Synapse.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Synapse.h"
// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>
// allow string comparison
#include <string.h>


// Macro to get the gaussian prob
//	INPUT:
//			x = The pre-population input neuron position that is being checked
//			u = The post-population neuron to which the connection is forming (taken as mean)
//			sigma = Standard Deviation of the gaussian distribution
#define GAUS(x,u,sigma) ( (1.0f/(sigma*(sqrt(2.0f*M_PI)))) * (exp(-1.0f * (pow((x-u),(2.0f))) / (2.0f*(pow(sigma,(2.0f)))))) )

// Synapse Constructor
Synapse::Synapse() {
	// Initialise my parameters
	// Variables;
	numconnections = 0;
	pre = NULL;
	post = NULL;
	// Full Matrices
	presyns = NULL;
	postsyns = NULL;
	weights = NULL;
	lastactive = NULL;
	delays = NULL;
	stdp = NULL;
	spikes = NULL;
	// STDP
	w_max = 60.0f;
	a_minus = -0.015f;
	a_plus = 0.005;
	tau_minus = 0.025;
	tau_plus = 0.015;

	// On construction, seed
	srand(42);	// Seeding the random numbers
}

// Synapse Destructor
Synapse::~Synapse() {
	// Just need to free up the memory
	free(pre);
	free(post);
	// Full Matrices
	free(presyns);
	free(postsyns);
	free(weights);
	free(lastactive);
	free(delays);
	free(stdp);
	free(spikes);
}

// Setting personal STDP parameters
void Synapse::SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new){
	// Set the values
	w_max = w_max_new;
	a_minus = a_minus_new;
	a_plus = a_plus_new;
	tau_minus = tau_minus_new;
	tau_plus = tau_plus_new;
}

// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		Style = Character List, e.g. "all_to_all"
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random connections or S.D. for Gaussian
void Synapse::AddConnection(	int pre, 
								int post, 
								int* popNums, 
								char style[], 
								float weightrange[2],
								int delayrange[2],
								bool stdpswitch,
								float parameter){
	// Find the right set of indices
	// Pre Indices
	int prestart = 0;
	if (pre > 0){
		prestart = popNums[pre-1];
	}
	int preend = popNums[pre];
	// Post Indices
	int poststart = 0;
	if (post > 0){
		poststart = popNums[post-1];
	}
	int postend = popNums[post];
	// Get the types of connections
	char option = 'w';
	if (strcmp(style, "all_to_all") == 0){
		option = 'a';
		int increment = (preend-prestart)*(postend-poststart);
		presyns = (int*)realloc(presyns, (numconnections + increment)*sizeof(int));
		postsyns = (int*)realloc(postsyns, (numconnections + increment)*sizeof(int));
		weights = (float*)realloc(weights, (numconnections + increment)*sizeof(float));
		lastactive = (float*)realloc(lastactive, (numconnections + increment)*sizeof(float));
		delays = (int*)realloc(delays, (numconnections + increment)*sizeof(int));
		stdp = (int*)realloc(stdp, (numconnections + increment)*sizeof(int));
		spikes = (int*)realloc(spikes, (numconnections + increment)*sizeof(int));
	} else if (strcmp(style, "one_to_one") == 0){
		option = 'o';
		int increment = (preend-prestart);
		presyns = (int*)realloc(presyns, (numconnections + increment)*sizeof(int));
		postsyns = (int*)realloc(postsyns, (numconnections + increment)*sizeof(int));
		weights = (float*)realloc(weights, (numconnections + increment)*sizeof(float));
		lastactive = (float*)realloc(lastactive, (numconnections + increment)*sizeof(float));
		delays = (int*)realloc(delays, (numconnections + increment)*sizeof(int));
		stdp = (int*)realloc(stdp, (numconnections + increment)*sizeof(int));
		spikes = (int*)realloc(spikes, (numconnections + increment)*sizeof(int));
	} else if (strcmp(style, "random") == 0){
		option = 'r';
	} else if (strcmp(style, "gaussian") == 0){
		option = 'g';
	} else {
		//Nothing
	}
	// Carry out the creation of the connectivity matrix
	switch (option){
		// ALL TO ALL
		case 'a':
			// If the connectivity is all_to_all
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Index
					int idx = numconnections + (i-prestart) + (j-poststart)*(preend-prestart);
					// Setup Synapses
					presyns[idx] = i;
					postsyns[idx] = j;
					// Setup Weights
					if (weightrange[0] == weightrange[1]) {
						weights[idx] = weightrange[0];
					} else {
						float rndweight = weightrange[0] + (weightrange[1] - weightrange[0])*((float)rand() / (RAND_MAX));
						weights[idx] = rndweight;
					}
					// Setup Delays
					// Get the randoms
					if (delayrange[0] == delayrange[1]) {
						delays[idx] = delayrange[0];
					} else {
						float rnddelay = delayrange[0] + (delayrange[1] - delayrange[0])*((float)rand() / (RAND_MAX));
						delays[idx] = round(rnddelay);
					}
					// Setup STDP
					if (stdpswitch){
						stdp[idx] = 1;
					} else {
						stdp[idx] = 0;
					}
					// Set Spikes
					spikes[idx] = 0;
				}
			}
			// Increment count
			numconnections += (preend-prestart)*(postend-poststart);
			break;
		// ONE TO ONE
		case 'o':
			// If the connectivity is one_to_one
			if ((preend-prestart) != (postend-poststart)){
				printf("Unequal populations for one_to_one");
				exit(-1);
			}
			// Create the connectivity
			for (int i = 0; i < (preend-prestart); i++){
				presyns[numconnections + i] = prestart + i;
				postsyns[numconnections + i] = poststart + i;
				// Setting up the weights
				if (weightrange[0] == weightrange[1]) {
					weights[numconnections + i] = weightrange[0];
				} else {
					float rndweight = weightrange[0] + (weightrange[1] - weightrange[0])*((float)rand() / (RAND_MAX));
					weights[numconnections + i] = rndweight;
				}
				// Setting up the delays
				if (delayrange[0] == delayrange[1]) {
					delays[numconnections + i] = delayrange[0];
				} else {
					float rnddelay = delayrange[0] + (delayrange[1] - delayrange[0])*((float)rand() / (RAND_MAX));
					delays[numconnections + i] = round(rnddelay);
				}
				// Setup STDP
				if (stdpswitch){
					stdp[numconnections + i] = 1;
				} else {
					stdp[numconnections + i] = 0;
				}
				// Set Spikes
				spikes[numconnections + i] = 0;
			}
			// Increment count
			numconnections += preend-prestart;
			break;
		// RANDOM
		case 'r':
			// If the connectivity is random
			// Begin a count
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Probability of connection
					float prob = ((float)rand() / (RAND_MAX));
					// If it is within the probability range, connect!
					if (prob < parameter){
						// Increase count
						numconnections += 1;
						presyns = (int*)realloc(presyns, (numconnections)*sizeof(int));
						postsyns = (int*)realloc(postsyns, (numconnections)*sizeof(int));
						weights = (float*)realloc(weights, (numconnections)*sizeof(float));
						lastactive = (float*)realloc(lastactive, (numconnections)*sizeof(float));
						delays = (int*)realloc(delays, (numconnections)*sizeof(int));
						stdp = (int*)realloc(stdp, (numconnections)*sizeof(int));
						spikes = (int*)realloc(spikes, (numconnections)*sizeof(int));
						// Setup Synapses
						presyns[numconnections - 1] = i;
						postsyns[numconnections - 1] = j;
						// Setup Weights
						if (weightrange[0] == weightrange[1]) {
							weights[numconnections - 1] = weightrange[0];
						} else {
							float rndweight = weightrange[0] + (weightrange[1] - weightrange[0])*((float)rand() / (RAND_MAX));
							weights[numconnections - 1] = rndweight;
						}
						// Setup Delays
						if (delayrange[0] == delayrange[1]) {
							delays[numconnections - 1] = delayrange[0];
						} else {
							float rnddelay = delayrange[0] + (delayrange[1] - delayrange[0])*((float)rand() / (RAND_MAX));
							delays[numconnections - 1] = round(rnddelay);
						}
						// Setup STDP
						if (stdpswitch){
							stdp[numconnections - 1] = 1;
						} else {
							stdp[numconnections - 1] = 0;
						}
						// Set Spikes
						spikes[numconnections - 1] = 0;
					}
				}
			}
			break;
		// GAUSSIAN
		case 'g':
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Probability of connection
					float prob = ((float) rand() / (RAND_MAX));
					// If it is within the probability range, connect!
					if (prob <= (GAUS((float)(i-prestart),(float)(j-poststart),parameter))){
						// Increase count
						numconnections += 1;
						presyns = (int*)realloc(presyns, (numconnections)*sizeof(int));
						postsyns = (int*)realloc(postsyns, (numconnections)*sizeof(int));
						weights = (float*)realloc(weights, (numconnections)*sizeof(float));
						lastactive = (float*)realloc(lastactive, (numconnections)*sizeof(float));
						delays = (int*)realloc(delays, (numconnections)*sizeof(int));
						stdp = (int*)realloc(stdp, (numconnections)*sizeof(int));
						spikes = (int*)realloc(spikes, (numconnections)*sizeof(int));
						// Setup Synapses
						presyns[numconnections - 1] = i;
						postsyns[numconnections - 1] = j;
						// Setup Weights
						if (weightrange[0] == weightrange[1]) {
							weights[numconnections - 1] = weightrange[0];
						} else {
							float rndweight = weightrange[0] + (weightrange[1] - weightrange[0])*((float)rand() / (RAND_MAX));
							weights[numconnections - 1] = rndweight;
						}
						// Setup Delays
						if (delayrange[0] == delayrange[1]) {
							delays[numconnections - 1] = delayrange[0];
						} else {
							float rnddelay = delayrange[0] + (delayrange[1] - delayrange[0])*((float)rand() / (RAND_MAX));
							delays[numconnections - 1] = round(rnddelay);
						}
						// Setup STDP
						if (stdpswitch){
							stdp[numconnections - 1] = 1;
						} else {
							stdp[numconnections - 1] = 0;
						}
						// Set Spikes
						spikes[numconnections - 1] = 0;
					}
				}
			}
			break;
		default:
			printf("\n\nUnknown Connection Type\n\n");
			exit(-1);
			break;
	}
}


