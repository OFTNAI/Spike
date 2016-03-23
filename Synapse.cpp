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

#include "Constants.h"


// Macro to get the gaussian prob
//	INPUT:
//			x = The pre-population input neuron position that is being checked
//			u = The post-population neuron to which the connection is forming (taken as mean)
//			sigma = Standard Deviation of the gaussian distribution
#define GAUS(distance, sigma) ( (1.0f/(sigma*(sqrt(2.0f*M_PI)))) * (exp(-1.0f * (pow((distance),(2.0f))) / (2.0f*(pow(sigma,(2.0f)))))) )

// Synapse Constructor
Synapse::Synapse() {
	// Initialise my parameters
	// Variables;
	numconnections = 0;
	// Full Matrices
	presyns = NULL;
	postsyns = NULL;
	weights = NULL;
	lastactive = NULL;
	delays = NULL;
	stdp = NULL;

	// On construction, seed
	srand(42);	// Seeding the random numbers
}

// Synapse Destructor
Synapse::~Synapse() {
	// Just need to free up the memory
	// Full Matrices
	free(presyns);
	free(postsyns);
	free(weights);
	free(lastactive);
	free(delays);
	free(stdp);
}

// Setting personal STDP parameters
void Synapse::SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new){
	// Set the values
	stdp_vars.w_max = w_max_new;
	stdp_vars.a_minus = a_minus_new;
	stdp_vars.a_plus = a_plus_new;
	stdp_vars.tau_minus = tau_minus_new;
	stdp_vars.tau_plus = tau_plus_new;
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
void Synapse::AddConnection(	int prepop, 
								int postpop, 
								int* popNums,
								int** pop_shapes, 
								int connectivity_type,
								float weightrange[2],
								int delayrange[2],
								bool stdpswitch,
								float parameter,
								float parameter_two){
	// Find the right set of indices
	// Take everything in 2D
	// Pre-Population Indices
	int prestart = 0;
	if (prepop > 0){
		prestart = popNums[prepop-1];
	}
	int preend = popNums[prepop];
	// Post-Population Indices
	int poststart = 0;
	if (postpop > 0){
		poststart = popNums[postpop-1];
	}
	int postend = popNums[postpop];
	// Get the types of connections
	char option = 'w';
	if (connectivity_type == CONNECTIVITY_TYPE_ALL_TO_ALL) {
		option = 'a';
		int increment = (preend-prestart)*(postend-poststart);
		presyns = (int*)realloc(presyns, (numconnections + increment)*sizeof(int));
		postsyns = (int*)realloc(postsyns, (numconnections + increment)*sizeof(int));
		weights = (float*)realloc(weights, (numconnections + increment)*sizeof(float));
		lastactive = (float*)realloc(lastactive, (numconnections + increment)*sizeof(float));
		delays = (int*)realloc(delays, (numconnections + increment)*sizeof(int));
		stdp = (int*)realloc(stdp, (numconnections + increment)*sizeof(int));
	} else if (connectivity_type == CONNECTIVITY_TYPE_ONE_TO_ONE){
		option = 'o';
		int increment = (preend-prestart);
		presyns = (int*)realloc(presyns, (numconnections + increment)*sizeof(int));
		postsyns = (int*)realloc(postsyns, (numconnections + increment)*sizeof(int));
		weights = (float*)realloc(weights, (numconnections + increment)*sizeof(float));
		lastactive = (float*)realloc(lastactive, (numconnections + increment)*sizeof(float));
		delays = (int*)realloc(delays, (numconnections + increment)*sizeof(int));
		stdp = (int*)realloc(stdp, (numconnections + increment)*sizeof(int));
	} else if (connectivity_type == CONNECTIVITY_TYPE_RANDOM) {
		option = 'r';
	} else if (connectivity_type == CONNECTIVITY_TYPE_GAUSSIAN) {
		option = 'g';
	} else if (connectivity_type == CONNECTIVITY_TYPE_IRINA_GAUSSIAN) {
		option = 'i';
	} else if (connectivity_type == CONNECTIVITY_TYPE_SINGLE) {
		option = 's';
	} else {
		//Nothing
	}
	// Carry out the creation of the connectivity matrix
	switch (option){
		// ALL TO ALL
		case 'a':
		{
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
				}
			}
			// Increment count
			numconnections += (preend-prestart)*(postend-poststart);
			break;
		}
		// ONE TO ONE
		case 'o':
		{
			// If the connectivity is one_to_one
			if ((preend-prestart) != (postend-poststart)){
				printf("Unequal populations for one_to_one. Exiting.\n");
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
			}
			// Increment count
			numconnections += preend-prestart;
			break;
		}
		// RANDOM
		case 'r':
		{
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
					}
				}
			}
			break;
		}
		// GAUSSIAN (1-D or 2-D)
		case 'g':
		{
			// For gaussian connectivity, the shape of the layers matters.
			// If we desire a given number of neurons, we must scale the gaussian
			float gaussian_scaling_factor = 1.0f;
			if (parameter_two != 0.0f){
				gaussian_scaling_factor = 0.0f;
				int pre_x = pop_shapes[prepop][0] / 2;
				int pre_y = pop_shapes[prepop][1] / 2;
				for (int i = 0; i < pop_shapes[postpop][0]; i++){
					for (int j = 0; j < pop_shapes[postpop][1]; j++){
						// Post XY
						int post_x = i;
						int post_y = j;
						// Distance
						float distance = pow((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)), 0.5f);
						// Gaussian Probability
						gaussian_scaling_factor += GAUS(distance, parameter);
					}
				}
				// Multiplying the gaussian scaling factor by the number of connections you require:
				gaussian_scaling_factor = gaussian_scaling_factor / parameter_two;
			}
			// Running through our neurons
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Probability of connection
					float prob = ((float) rand() / (RAND_MAX));
					// Get the relative distance from the two neurons
					// Pre XY
					int pre_x = (i-prestart) % pop_shapes[prepop][0];
					int pre_y = floor((float)(i-prestart) / pop_shapes[prepop][0]);
					// Post XY
					int post_x = (j-poststart) % pop_shapes[postpop][0];
					int post_y = floor((float)(j-poststart) / pop_shapes[postpop][0]);
					// Distance
					float distance = sqrt((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)));
					// If it is within the probability range, connect!
					if (prob <= ((GAUS(distance, parameter)) / gaussian_scaling_factor)){
						// Increase count
						numconnections += 1;
						presyns = (int*)realloc(presyns, (numconnections)*sizeof(int));
						postsyns = (int*)realloc(postsyns, (numconnections)*sizeof(int));
						weights = (float*)realloc(weights, (numconnections)*sizeof(float));
						lastactive = (float*)realloc(lastactive, (numconnections)*sizeof(float));
						delays = (int*)realloc(delays, (numconnections)*sizeof(int));
						stdp = (int*)realloc(stdp, (numconnections)*sizeof(int));
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
					}
				}
			}
			break;
		}
		// IRINA's GAUSSIAN (1-D only)
		case 'i': 
		{
			// Getting the population sizes
			int in_size = preend - prestart;
			int out_size = postend - poststart;
			// Diagonal Width value
			float diagonal_width = parameter;
			// Irina's application of some sparse measure
			float in2out_sparse = 0.67f*0.67f;
			// Irina's implementation of some kind of stride
			int dist = 1;
			if ( (float(out_size)/float(in_size)) > 1.0f ){
				dist = int(out_size/in_size);
			}
			// Irina's version of sigma
			double sigma = dist*diagonal_width;
			// Number of connections to form
			int conn_num = int((sigma/in2out_sparse));
			int conn_tgts = 0;
			int temp = 0;
			// Running through the input neurons
			for (int i = prestart; i < preend; i++){
				double mu = int(float(dist)/2.0f) + (i-prestart)*dist;
				conn_tgts = 0;
				while (conn_tgts < conn_num) {
					temp = int(randn(mu, sigma));
					if ((temp >= 0) && (temp < out_size)){
						// Create space for a connection
						numconnections += 1;
						presyns = (int*)realloc(presyns, (numconnections)*sizeof(int));
						postsyns = (int*)realloc(postsyns, (numconnections)*sizeof(int));
						weights = (float*)realloc(weights, (numconnections)*sizeof(float));
						lastactive = (float*)realloc(lastactive, (numconnections)*sizeof(float));
						delays = (int*)realloc(delays, (numconnections)*sizeof(int));
						stdp = (int*)realloc(stdp, (numconnections)*sizeof(int));
						// Setup the synapses:
						// Setup Synapses
						presyns[numconnections - 1] = i;
						postsyns[numconnections - 1] = poststart + temp;
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
						// Increment conn_tgts
						++conn_tgts;
					}
				}
			}
			break;
		}
		// SINGLE Connection
		case 's':
		{
			// If we desire a single connection
			// Increase count
			numconnections += 1;
			presyns = (int*)realloc(presyns, (numconnections)*sizeof(int));
			postsyns = (int*)realloc(postsyns, (numconnections)*sizeof(int));
			weights = (float*)realloc(weights, (numconnections)*sizeof(float));
			lastactive = (float*)realloc(lastactive, (numconnections)*sizeof(float));
			delays = (int*)realloc(delays, (numconnections)*sizeof(int));
			stdp = (int*)realloc(stdp, (numconnections)*sizeof(int));
			// Setup Synapses
			presyns[numconnections - 1] = prestart + int(parameter);
			postsyns[numconnections - 1] = poststart + int(parameter_two);
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
			break;
		}
		default:
		{
			printf("\n\nUnknown Connection Type\n\n");
			exit(-1);
			break;
		}
	}
}





// An implementation of the polar gaussian random number generator which I need
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}