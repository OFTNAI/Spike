//	Synapse Class C++
//	Synapse.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Synapses.h"
// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>
#include "../Helpers/CUDAErrorCheckHelpers.h"


// Macro to get the gaussian prob
//	INPUT:
//			x = The pre-population input neuron position that is being checked
//			u = The post-population neuron to which the connection is forming (taken as mean)
//			sigma = Standard Deviation of the gaussian distribution
#define GAUS(distance, sigma) ( (1.0f/(sigma*(sqrt(2.0f*M_PI)))) * (exp(-1.0f * (pow((distance),(2.0f))) / (2.0f*(pow(sigma,(2.0f)))))) )

// Synapses Constructor
Synapses::Synapses() {
	// Initialise my parameters
	// Variables;
	total_number_of_synapses = 0;
	// Full Matrices
	presynaptic_neuron_indices = NULL;
	postsynaptic_neuron_indices = NULL;
	weights = NULL;
	delays = NULL;
	stdp = NULL;

	d_presynaptic_neuron_indices = NULL;
	d_postsynaptic_neuron_indices = NULL;
	d_delays = NULL;
	d_weights = NULL;
	d_spikes = NULL;
	d_stdp = NULL;
	d_lastactive = NULL;
	d_spikebuffer = NULL;

	// On construction, seed
	srand(42);	// Seeding the random numbers
}

// Synapses Destructor
Synapses::~Synapses() {
	// Just need to free up the memory
	// Full Matrices
	free(presynaptic_neuron_indices);
	free(postsynaptic_neuron_indices);
	free(weights);
	free(delays);
	free(stdp);

	CudaSafeCall(cudaFree(d_presynaptic_neuron_indices));
	CudaSafeCall(cudaFree(d_postsynaptic_neuron_indices));
	CudaSafeCall(cudaFree(d_delays));
	CudaSafeCall(cudaFree(d_weights));
	CudaSafeCall(cudaFree(d_spikes));
	CudaSafeCall(cudaFree(d_stdp));
	CudaSafeCall(cudaFree(d_lastactive));
	CudaSafeCall(cudaFree(d_spikebuffer));

}

// Setting personal STDP parameters
void Synapses::SetSTDP(float w_max_new,
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
//		Parameter = either probability for random synapses or S.D. for Gaussian
void Synapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two) {
	
	// Find the right set of indices
	// Take everything in 2D

	printf("presynaptic_group_id: %d\n", presynaptic_group_id);
	printf("postsynaptic_group_id: %d\n", postsynaptic_group_id);

	int* last_neuron_indices_for_neuron_groups = neurons->last_neuron_indices_for_each_group;
	int* last_neuron_indices_for_input_neuron_groups = input_neurons->last_neuron_indices_for_each_group;
	int** neuron_group_shapes = neurons->group_shapes; //OLD

	int * presynaptic_group_shape;
	int * postsynaptic_group_shape;

	int group_type_factor = 1;
	int group_type_component = 0;
	int prestart = 0;
	int preend = 0;
	int poststart = 0;

	// Calculate presynaptic group start and end indices
	// Also assign presynaptic group shape
	if (presynaptic_group_id < 0) { // If presynaptic group is Input group

		if (stdp_on == true) {
			printf("Plasticity between input neurons and model neurons is not currently supported. Exiting...\n");
			exit(-1);
		}

		group_type_factor = -1;
		group_type_component = -1;
		presynaptic_group_shape = input_neurons->group_shapes[-1*presynaptic_group_id - 1];

		if (presynaptic_group_id < -1){
			prestart = last_neuron_indices_for_input_neuron_groups[-1*presynaptic_group_id - 2];
		}
		preend = last_neuron_indices_for_input_neuron_groups[-1*presynaptic_group_id - 1];

	} else {

		presynaptic_group_shape = neurons->group_shapes[presynaptic_group_id];

		if (presynaptic_group_id > 0){
			prestart = last_neuron_indices_for_neuron_groups[presynaptic_group_id - 1];
		}
		preend = last_neuron_indices_for_neuron_groups[presynaptic_group_id];

	}

	// Calculate postsynaptic group start and end indices
	// Also assign postsynaptic group shape
	if (postsynaptic_group_id < 0) { // If presynaptic group is Input group EXIT

		printf("Input groups cannot be a postsynaptic neuron group. Exiting...\n");
		exit(-1);

	} else if (postsynaptic_group_id >= 0){
		postsynaptic_group_shape = neurons->group_shapes[postsynaptic_group_id];

		poststart = last_neuron_indices_for_neuron_groups[postsynaptic_group_id-1];
	}
	int postend = last_neuron_indices_for_neuron_groups[postsynaptic_group_id];


	printf("prestart: %d\n", prestart);
	printf("preend: %d\n", preend);
	printf("poststart: %d\n", poststart);
	printf("postend: %d\n\n", postend);


	int original_number_of_synapses = total_number_of_synapses;

	// Carry out the creation of the connectivity matrix
	switch (connectivity_type){
            
		case CONNECTIVITY_TYPE_ALL_TO_ALL:
		{
            
            int increment = (preend-prestart)*(postend-poststart);
            increment_number_of_synapses(increment);
            
			// If the connectivity is all_to_all
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Index
					int idx = original_number_of_synapses + (i-prestart) + (j-poststart)*(preend-prestart);
					// Setup Synapses
					presynaptic_neuron_indices[idx] = group_type_factor*i + group_type_component;
					postsynaptic_neuron_indices[idx] = j;
				}
			}
			break;
		}
		case CONNECTIVITY_TYPE_ONE_TO_ONE:
		{
            int increment = (preend-prestart);
            increment_number_of_synapses(increment);
            
			// If the connectivity is one_to_one
			if ((preend-prestart) != (postend-poststart)){
				printf("Unequal populations for one_to_one. Exiting...\n");
				exit(-1);
			}
			// Create the connectivity
			for (int i = 0; i < (preend-prestart); i++){
				presynaptic_neuron_indices[original_number_of_synapses + i] = group_type_factor*(prestart + i) + group_type_component;
				postsynaptic_neuron_indices[original_number_of_synapses + i] = poststart + i;
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
						
						increment_number_of_synapses(1);

						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_synapses - 1] = group_type_factor*i + group_type_component;
						postsynaptic_neuron_indices[total_number_of_synapses - 1] = j;
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
				int pre_x = neuron_group_shapes[presynaptic_group_id][0] / 2;
				int pre_y = neuron_group_shapes[presynaptic_group_id][1] / 2;
				for (int i = 0; i < neuron_group_shapes[postsynaptic_group_id][0]; i++){
					for (int j = 0; j < neuron_group_shapes[postsynaptic_group_id][1]; j++){
						// Post XY
						int post_x = i;
						int post_y = j;
						// Distance
						float distance = pow((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)), 0.5f);
						// Gaussian Probability
						gaussian_scaling_factor += GAUS(distance, parameter);
					}
				}
				// Multiplying the gaussian scaling factor by the number of synapses you require:
				gaussian_scaling_factor = gaussian_scaling_factor / parameter_two;
			}
			// Running through our neurons
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Probability of connection
					float prob = ((float) rand() / (RAND_MAX));
					// Get the relative distance from the two neurons
					// Pre XY
					int pre_x = (i-prestart) % neuron_group_shapes[presynaptic_group_id][0];
					int pre_y = floor((float)(i-prestart) / neuron_group_shapes[presynaptic_group_id][0]);
					// Post XY
					int post_x = (j-poststart) % neuron_group_shapes[postsynaptic_group_id][0];
					int post_y = floor((float)(j-poststart) / neuron_group_shapes[postsynaptic_group_id][0]);
					// Distance
					float distance = sqrt((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)));
					// If it is within the probability range, connect!
					if (prob <= ((GAUS(distance, parameter)) / gaussian_scaling_factor)){
						
						increment_number_of_synapses(1);

						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_synapses - 1] = group_type_factor*i + group_type_component;
						postsynaptic_neuron_indices[total_number_of_synapses - 1] = j;
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
			// Number of synapses to form
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
						
						increment_number_of_synapses(1);

						// Setup the synapses:
						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_synapses - 1] = group_type_factor*i + group_type_component;
						postsynaptic_neuron_indices[total_number_of_synapses - 1] = poststart + temp;

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
			increment_number_of_synapses(1);

			// Setup Synapses
			presynaptic_neuron_indices[original_number_of_synapses] = group_type_factor * (prestart + int(parameter)) + group_type_component;
			postsynaptic_neuron_indices[original_number_of_synapses] = poststart + int(parameter_two);

			break;
		}
		default:
		{
			printf("\n\nUnknown Connection Type: %d\n\n", connectivity_type);
			exit(-1);
			break;
		}
	}

	for (int i = original_number_of_synapses; i < total_number_of_synapses-1; i++){
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

void Synapses::increment_number_of_synapses(int increment) {
	printf("Increment: %d\n", increment);
	presynaptic_neuron_indices = (int*)realloc(presynaptic_neuron_indices, (total_number_of_synapses + increment)*sizeof(int));
    postsynaptic_neuron_indices = (int*)realloc(postsynaptic_neuron_indices, (total_number_of_synapses + increment)*sizeof(int));
    weights = (float*)realloc(weights, (total_number_of_synapses + increment)*sizeof(float));
    delays = (int*)realloc(delays, (total_number_of_synapses + increment)*sizeof(int));
    stdp = (int*)realloc(stdp, (total_number_of_synapses + increment)*sizeof(int));

    total_number_of_synapses += increment;
}


void Synapses::initialise_device_pointers() {
	CudaSafeCall(cudaMalloc((void **)&d_presynaptic_neuron_indices, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_postsynaptic_neuron_indices, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_delays, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_weights, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikes, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_stdp, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_lastactive, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_spikebuffer, sizeof(int)*total_number_of_synapses));


	CudaSafeCall(cudaMemcpy(d_presynaptic_neuron_indices, presynaptic_neuron_indices, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsynaptic_neuron_indices, postsynaptic_neuron_indices, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_delays, delays, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_weights, weights, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_stdp, stdp, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));

	reset_synapse_spikes();
}

void Synapses::reset_synapse_spikes() {
	CudaSafeCall(cudaMemset(d_spikes, 0, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_lastactive, -1000.0f, sizeof(float)*total_number_of_synapses));
	CudaSafeCall(cudaMemset(d_spikebuffer, -1, sizeof(int)*total_number_of_synapses));
}


void Synapses::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

	int number_of_synapse_blocks = (total_number_of_synapses + threads) / threads;
	number_of_synapse_blocks_per_grid.x = number_of_synapse_blocks;

	printf("number_of_synapse_blocks_per_grid.x: %d\n", number_of_synapse_blocks_per_grid.x);
}



__global__ void calculate_postsynaptic_current_injection_for_synapse_kernal(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses);

__global__ void check_for_synapse_spike_arrival_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								int* d_spikebuffer,
								float currtime,
								size_t total_number_of_synapses);

__global__ void apply_ltd_to_synapse_weights_kernal(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t numConns);

__global__ void apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns);



void Synapses::calculate_postsynaptic_current_injection_for_synapse(float* d_neurons_current_injections, float current_time_in_seconds) {

	calculate_postsynaptic_current_injection_for_synapse_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_spikes,
																	d_weights,
																	d_lastactive,
																	d_postsynaptic_neuron_indices,
																	d_neurons_current_injections,
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}

void Synapses::check_for_synapse_spike_arrival(float* d_neurons_last_spike_time, float* d_input_neurons_last_spike_time, float current_time_in_seconds) {

	check_for_synapse_spike_arrival_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																		d_delays,
																		d_spikes,
																		d_neurons_last_spike_time,
																		d_input_neurons_last_spike_time,
																		d_spikebuffer,
																		current_time_in_seconds,
																		total_number_of_synapses);

	CudaCheckError();
}

void Synapses::apply_ltd_to_synapse_weights(float* d_lastspiketime, float current_time_in_seconds) {

	apply_ltd_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_lastactive,
																	d_weights,
																	d_stdp,
																	d_lastspiketime,
																	d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	stdp_vars, // Should make device copy?
																	total_number_of_synapses);

	CudaCheckError();
}


void Synapses::apply_ltp_to_synapse_weights(float* d_lastspiketime, float current_time_in_seconds) {
	// Carry out the last step, LTP!
	apply_ltp_to_synapse_weights_kernal<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_postsynaptic_neuron_indices,
																	d_lastspiketime,
																	d_stdp,
																	d_lastactive,
																	d_weights,
																	stdp_vars, 
																	current_time_in_seconds,
																	total_number_of_synapses);

	CudaCheckError();
}



// If spike has reached synapse add synapse weight to postsyn current injection
// Was currentcalc
__global__ void calculate_postsynaptic_current_injection_for_synapse_kernal(int* d_spikes,
							float* d_weights,
							float* d_lastactive,
							int* d_postsynaptic_neuron_indices,
							float* d_neurons_current_injections,
							float current_time_in_seconds,
							size_t total_number_of_synapses){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (total_number_of_synapses)) {
		// Decrememnt Spikes
		d_spikes[idx] -= 1;
		if (d_spikes[idx] == 0) {

			atomicAdd(&d_neurons_current_injections[d_postsynaptic_neuron_indices[idx]], d_weights[idx]);

			// Change lastactive
			d_lastactive[idx] = current_time_in_seconds;
			// Done!
		}
	}
	__syncthreads();
}


// Synapses carrying spikes
// JI GIVE A BETTER NAME
__global__ void check_for_synapse_spike_arrival_kernal(int* d_presynaptic_neuron_indices,
								int* d_delays,
								int* d_spikes,
								float* d_neurons_last_spike_time,
								float* d_input_neurons_last_spike_time,
								int* d_spikebuffer,
								float current_time_in_seconds,
								size_t total_number_of_synapses){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_synapses) {
		// Reduce the spikebuffer by 1
		d_spikebuffer[idx] -= 1;

		int presynaptic_neuron_index = d_presynaptic_neuron_indices[idx];
		float presynaptic_neurons_last_spike_time;
		if (presynaptic_neuron_index < 0) {
			presynaptic_neurons_last_spike_time = d_input_neurons_last_spike_time[-1*presynaptic_neuron_index - 1];
		} else {
			presynaptic_neurons_last_spike_time = d_neurons_last_spike_time[presynaptic_neuron_index];
		}

		// Check if the neuron PRE has just fired and if the synapse exists
		if (presynaptic_neurons_last_spike_time == current_time_in_seconds){
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



// LTD of weights
__global__ void apply_ltd_to_synapse_weights_kernal(float* d_lastactive,
							float* d_weights,
							int* d_stdp,
							float* d_lastspiketime,
							int* d_postsyns,
							float currtime,
							struct stdp_struct stdp_vars,
							size_t numConns){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < (numConns)) {
		// Get the locations for updating
		// Get the synapses that are to be LTD'd
		if ((d_lastactive[idx] == currtime) && (d_stdp[idx] == 1)) {
			float diff = d_lastspiketime[d_postsyns[idx]] - currtime;
			// STDP Update Rule
			float weightscale = stdp_vars.w_max * stdp_vars.a_minus * expf(diff / stdp_vars.tau_minus);
			// Now scale the weight (using an inverted column/row)
			d_weights[idx] += weightscale; 
		}
	}
}


// LTP on synapses
__global__ void apply_ltp_to_synapse_weights_kernal(int* d_postsyns,
							float* d_lastspiketime,
							int* d_stdp,
							float* d_lastactive,
							float* d_weights,
							struct stdp_struct stdp_vars,
							float currtime,
							size_t numConns) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numConns) {
		// Get the synapses upon which we should do LTP
		// Reversed indexing to check post->pre synapses
		if ((d_lastspiketime[d_postsyns[idx]] == currtime) && (d_stdp[idx] == 1)){
			// Get the last active time / weight of the synapse
			// Calc time difference and weight change
			float diff = currtime - d_lastactive[idx];
			float weightchange = (stdp_vars.w_max - d_weights[idx]) * (stdp_vars.a_plus * expf(-diff / stdp_vars.tau_plus));
			// Update weights
			d_weights[idx] += weightchange;
		}

	}
}


__device__ int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
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