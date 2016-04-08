//	Synapse Class C++
//	Synapse.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Connections.h"
// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include "Constants.h"
#include <cuda.h>
#include "CUDAErrorCheckHelpers.h"


// Macro to get the gaussian prob
//	INPUT:
//			x = The pre-population input neuron position that is being checked
//			u = The post-population neuron to which the connection is forming (taken as mean)
//			sigma = Standard Deviation of the gaussian distribution
#define GAUS(distance, sigma) ( (1.0f/(sigma*(sqrt(2.0f*M_PI)))) * (exp(-1.0f * (pow((distance),(2.0f))) / (2.0f*(pow(sigma,(2.0f)))))) )

// Connections Constructor
Connections::Connections() {
	// Initialise my parameters
	// Variables;
	total_number_of_connections = 0;
	// Full Matrices
	presynaptic_neuron_indices = NULL;
	postsynaptic_neuron_indices = NULL;
	weights = NULL;
	delays = NULL;
	stdp = NULL;

	// On construction, seed
	srand(42);	// Seeding the random numbers
}

// Connections Destructor
Connections::~Connections() {
	// Just need to free up the memory
	// Full Matrices
	free(presynaptic_neuron_indices);
	free(postsynaptic_neuron_indices);
	free(weights);
	free(delays);
	free(stdp);
}

// Setting personal STDP parameters
void Connections::SetSTDP(float w_max_new,
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
//		CONNECTIVITY_TYPE (Constants.h)
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random connections or S.D. for Gaussian
void Connections::AddGroup(	int presynaptic_group_id, 
								int postsynaptic_group_id, 
								int* last_neuron_indices_for_each_neuron_group,
								int** group_shapes, 
								int connectivity_type,
								float weight_range[2],
								int delay_range[2],
								bool stdp_on,
								float parameter,
								float parameter_two){
	// Find the right set of indices
	// Take everything in 2D
	// Pre-Population Indices
	int prestart = 0;
	if (presynaptic_group_id > 0){
		prestart = last_neuron_indices_for_each_neuron_group[presynaptic_group_id-1];
		printf("prestart: %d\n", prestart);
	}
	int preend = last_neuron_indices_for_each_neuron_group[presynaptic_group_id];
	printf("preend: %d\n", preend);
	// Post-Population Indices
	int poststart = 0;
	if (postsynaptic_group_id > 0){
		poststart = last_neuron_indices_for_each_neuron_group[postsynaptic_group_id-1];
	}
	int postend = last_neuron_indices_for_each_neuron_group[postsynaptic_group_id];

	int original_number_of_connections = total_number_of_connections;

	// Carry out the creation of the connectivity matrix
	switch (connectivity_type){
            
		case CONNECTIVITY_TYPE_ALL_TO_ALL:
		{
            
            int increment = (preend-prestart)*(postend-poststart);
            increment_number_of_connections(increment);
            
			// If the connectivity is all_to_all
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Index
					int idx = original_number_of_connections + (i-prestart) + (j-poststart)*(preend-prestart);
					// Setup Synapses
					presynaptic_neuron_indices[idx] = i;
					postsynaptic_neuron_indices[idx] = j;
				}
			}
			break;
		}
		case CONNECTIVITY_TYPE_ONE_TO_ONE:
		{
            int increment = (preend-prestart);
            increment_number_of_connections(increment);
            
			// If the connectivity is one_to_one
			if ((preend-prestart) != (postend-poststart)){
				printf("Unequal populations for one_to_one. Exiting.\n");
				exit(-1);
			}
			// Create the connectivity
			for (int i = 0; i < (preend-prestart); i++){
				presynaptic_neuron_indices[original_number_of_connections + i] = prestart + i;
				postsynaptic_neuron_indices[original_number_of_connections + i] = poststart + i;
			}

			break;
		}
		case CONNECTIVITY_TYPE_RANDOM: //JI DO
		{
			// If the connectivity is random
			// Begin a count
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Probability of connection
					float prob = ((float)rand() / (RAND_MAX));
					// If it is within the probability range, connect!
					if (prob < parameter){
						
						increment_number_of_connections(1);

						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_connections - 1] = i;
						postsynaptic_neuron_indices[total_number_of_connections - 1] = j;
					}
				}
			}
			break;
		}
		
		case CONNECTIVITY_TYPE_GAUSSIAN: // 1-D or 2-D
		{
			// For gaussian connectivity, the shape of the layers matters.
			// If we desire a given number of neurons, we must scale the gaussian
			float gaussian_scaling_factor = 1.0f;
			if (parameter_two != 0.0f){
				gaussian_scaling_factor = 0.0f;
				int pre_x = group_shapes[presynaptic_group_id][0] / 2;
				int pre_y = group_shapes[presynaptic_group_id][1] / 2;
				for (int i = 0; i < group_shapes[postsynaptic_group_id][0]; i++){
					for (int j = 0; j < group_shapes[postsynaptic_group_id][1]; j++){
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
					int pre_x = (i-prestart) % group_shapes[presynaptic_group_id][0];
					int pre_y = floor((float)(i-prestart) / group_shapes[presynaptic_group_id][0]);
					// Post XY
					int post_x = (j-poststart) % group_shapes[postsynaptic_group_id][0];
					int post_y = floor((float)(j-poststart) / group_shapes[postsynaptic_group_id][0]);
					// Distance
					float distance = sqrt((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)));
					// If it is within the probability range, connect!
					if (prob <= ((GAUS(distance, parameter)) / gaussian_scaling_factor)){
						
						increment_number_of_connections(1);

						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_connections - 1] = i;
						postsynaptic_neuron_indices[total_number_of_connections - 1] = j;
					}
				}
			}
			break;
		}
		case CONNECTIVITY_TYPE_IRINA_GAUSSIAN: // 1-D only
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
						
						increment_number_of_connections(1);

						// Setup the synapses:
						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_connections - 1] = i;
						postsynaptic_neuron_indices[total_number_of_connections - 1] = poststart + temp;

						// Increment conn_tgts
						++conn_tgts;
					}
				}
			}
			break;
		}
		case CONNECTIVITY_TYPE_SINGLE:
		{
			// If we desire a single connection
			increment_number_of_connections(1);

			// Setup Synapses
			presynaptic_neuron_indices[original_number_of_connections] = prestart + int(parameter);
			postsynaptic_neuron_indices[original_number_of_connections] = poststart + int(parameter_two);

			break;
		}
		default:
		{
			printf("\n\nUnknown Connection Type: %d\n\n", CONNECTIVITY_TYPE_SINGLE);
			exit(-1);
			break;
		}
	}

	for (int i = original_number_of_connections; i < total_number_of_connections-1; i++){
		// Setup Weights
		if (weight_range[0] == weight_range[1]) {
			weights[i] = weight_range[0];
		} else {
			float rndweight = weight_range[0] + (weight_range[1] - weight_range[0])*((float)rand() / (RAND_MAX));
			weights[i] = rndweight;
		}
		// Setup Delays
		// Get the randoms
		if (delay_range[0] == delay_range[1]) {
			delays[i] = delay_range[0];
		} else {
			float rnddelay = delay_range[0] + (delay_range[1] - delay_range[0])*((float)rand() / (RAND_MAX));
			delays[i] = round(rnddelay);
		}
		// Setup STDP
		if (stdp_on){
			stdp[i] = 1;
		} else {
			stdp[i] = 0;
		}
	}

}

void Connections::increment_number_of_connections(int increment) {
	presynaptic_neuron_indices = (int*)realloc(presynaptic_neuron_indices, (total_number_of_connections + increment)*sizeof(int));
    postsynaptic_neuron_indices = (int*)realloc(postsynaptic_neuron_indices, (total_number_of_connections + increment)*sizeof(int));
    weights = (float*)realloc(weights, (total_number_of_connections + increment)*sizeof(float));
    delays = (int*)realloc(delays, (total_number_of_connections + increment)*sizeof(int));
    stdp = (int*)realloc(stdp, (total_number_of_connections + increment)*sizeof(int));

    total_number_of_connections += increment;
}


void Connections::initialise_device_pointers() {
	CudaSafeCall(cudaMalloc((void **)&d_presynaptic_neuron_indices, sizeof(int)*total_number_of_connections));
	CudaSafeCall(cudaMalloc((void **)&d_postsynaptic_neuron_indices, sizeof(int)*total_number_of_connections));
	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*total_number_of_connections));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*total_number_of_connections));
	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*total_number_of_connections));


	CudaSafeCall(cudaMemcpy(d_presynaptic_neuron_indices, presynaptic_neuron_indices, sizeof(int)*total_number_of_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsynaptic_neuron_indices, postsynaptic_neuron_indices, sizeof(int)*total_number_of_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*total_number_of_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, weights, sizeof(float)*total_number_of_connections, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(d_spikes, 0, sizeof(int)*total_number_of_connections));
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