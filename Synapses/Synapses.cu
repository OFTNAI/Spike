//	Synapse Class C++
//	Synapse.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Synapses.h"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"

#include <algorithm> // for random shuffle

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>

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
	temp_number_of_synapses_in_last_group = 0;

	largest_synapse_group_size = 0;
	old_largest_number_of_blocks_x = 0;

	neuron_indices_set_up_on_device = false;

	original_synapse_indices = NULL;

	// Full Matrices
	presynaptic_neuron_indices = NULL;
	postsynaptic_neuron_indices = NULL;
	synaptic_efficacies_or_weights = NULL;

	d_temp_presynaptic_neuron_indices = NULL;
	d_temp_postsynaptic_neuron_indices = NULL;
	d_temp_synaptic_efficacies_or_weights = NULL;

	d_presynaptic_neuron_indices = NULL;
	d_postsynaptic_neuron_indices = NULL;
	d_synaptic_efficacies_or_weights = NULL;

	d_states_for_random_number_generation = NULL;

	random_state_manager = NULL;

	print_synapse_group_details = false;

	// On construction, seed
	srand(42);	// Seeding the random numbers
}

// Synapses Destructor
Synapses::~Synapses() {
	// Just need to free up the memory
	// Full Matrices
	free(presynaptic_neuron_indices);
	free(postsynaptic_neuron_indices);
	free(synaptic_efficacies_or_weights);

	CudaSafeCall(cudaFree(d_presynaptic_neuron_indices));
	CudaSafeCall(cudaFree(d_postsynaptic_neuron_indices));
	CudaSafeCall(cudaFree(d_synaptic_efficacies_or_weights));

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
						synapse_parameters_struct * synapse_params,
						float parameter,
						float parameter_two) {
	
	// Find the right set of indices
	// Take everything in 2D

	if (print_synapse_group_details == true) {
		printf("Adding synapse group...\n");
		printf("presynaptic_group_id: %d\n", presynaptic_group_id);
		printf("postsynaptic_group_id: %d\n", postsynaptic_group_id);
	}

	int* last_neuron_indices_for_neuron_groups = neurons->last_neuron_indices_for_each_group;
	int* last_neuron_indices_for_input_neuron_groups = input_neurons->last_neuron_indices_for_each_group;

	int * presynaptic_group_shape;
	int * postsynaptic_group_shape;

	int prestart = 0;
	int preend = 0;
	int poststart = 0;

	// Calculate presynaptic group start and end indices
	// Also assign presynaptic group shape
	bool presynaptic_group_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_group_id);

	if (presynaptic_group_is_input) {

		// if (stdp_on == true) print_message_and_exit("Plasticity between input neurons and model neurons is not currently supported.");

		presynaptic_group_shape = input_neurons->group_shapes[CORRECTED_PRESYNAPTIC_ID(presynaptic_group_id, presynaptic_group_is_input)];

		if (presynaptic_group_id < -1){
			prestart = last_neuron_indices_for_input_neuron_groups[CORRECTED_PRESYNAPTIC_ID(presynaptic_group_id, presynaptic_group_is_input) - 1];
		}
		preend = last_neuron_indices_for_input_neuron_groups[CORRECTED_PRESYNAPTIC_ID(presynaptic_group_id, presynaptic_group_is_input)];

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

		print_message_and_exit("Input groups cannot be a postsynaptic neuron group.");

	} else if (postsynaptic_group_id >= 0){
		postsynaptic_group_shape = neurons->group_shapes[postsynaptic_group_id];

		if (postsynaptic_group_id == 0) {
			poststart = 0;
		} else {
			poststart = last_neuron_indices_for_neuron_groups[postsynaptic_group_id - 1];
		}
		
	}
	int postend = last_neuron_indices_for_neuron_groups[postsynaptic_group_id];

	if (print_synapse_group_details == true) {
		const char * presynaptic_group_type_string = (presynaptic_group_id < 0) ? "input_neurons" : "neurons";
		printf("Presynaptic neurons start index: %d (%s)\n", prestart, presynaptic_group_type_string);
		printf("Presynaptic neurons end index: %d (%s)\n", preend, presynaptic_group_type_string);
		printf("Postsynaptic neurons start index: %d (neurons)\n", poststart);
		printf("Postsynaptic neurons end index: %d (neurons)\n", postend);
	}


	int original_number_of_synapses = total_number_of_synapses;

	// Carry out the creation of the connectivity matrix
	switch (connectivity_type){
            
		case CONNECTIVITY_TYPE_ALL_TO_ALL:
		{
            
            int increment = (preend-prestart)*(postend-poststart);
            this->increment_number_of_synapses(increment);
            
			// If the connectivity is all_to_all
			for (int i = prestart; i < preend; i++){
				for (int j = poststart; j < postend; j++){
					// Index
					int idx = original_number_of_synapses + (i-prestart) + (j-poststart)*(preend-prestart);
					// Setup Synapses
					presynaptic_neuron_indices[idx] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
					postsynaptic_neuron_indices[idx] = j;
				}
			}
			break;
		}
		case CONNECTIVITY_TYPE_ONE_TO_ONE:
		{
            int increment = (preend-prestart);
            this->increment_number_of_synapses(increment);
            
			// If the connectivity is one_to_one
			if ((preend-prestart) != (postend-poststart)) print_message_and_exit("Unequal populations for one_to_one.");
			// Create the connectivity
			for (int i = 0; i < (preend-prestart); i++){
				presynaptic_neuron_indices[original_number_of_synapses + i] = CORRECTED_PRESYNAPTIC_ID(prestart + i, presynaptic_group_is_input);
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
						
						this->increment_number_of_synapses(1);

						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_synapses - 1] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
						postsynaptic_neuron_indices[total_number_of_synapses - 1] = j;
					}
				}
			}
			break;
		}
		
		case CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE:
		{

			// neuron_indices_set_up_on_device = true; //Temp (Would introduce bugs if mixing connectivity type)

			float standard_deviation_sigma = parameter;

			int number_of_new_synapses_per_postsynaptic_neuron = 50;

			int number_of_postsynaptic_neurons_in_group = postend - poststart;

			int total_number_of_new_synapses = number_of_new_synapses_per_postsynaptic_neuron * number_of_postsynaptic_neurons_in_group;

			this->increment_number_of_synapses(total_number_of_new_synapses);




			//Setting up random states

			if (random_state_manager == NULL) {

				random_state_manager = new RandomStateManager();
				int threads_per_block_x = 128;
				int number_of_blocks_x = 64;
				random_state_manager->set_up_random_states(threads_per_block_x, number_of_blocks_x, 9);
				CudaCheckError();

			}

			if (total_number_of_new_synapses > largest_synapse_group_size) {

				largest_synapse_group_size = total_number_of_new_synapses;

				CudaSafeCall(cudaMalloc((void **)&d_temp_presynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses));
				CudaSafeCall(cudaMalloc((void **)&d_temp_postsynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses));

			}

			set_neuron_indices_by_sampling_from_normal_distribution<<<random_state_manager->block_dimensions, random_state_manager->threads_per_block>>>(total_number_of_new_synapses, postsynaptic_group_id, poststart, prestart, postsynaptic_group_shape[0], postsynaptic_group_shape[1], presynaptic_group_shape[0], presynaptic_group_shape[1], number_of_new_synapses_per_postsynaptic_neuron, number_of_postsynaptic_neurons_in_group, d_temp_presynaptic_neuron_indices, d_temp_postsynaptic_neuron_indices, d_temp_synaptic_efficacies_or_weights, standard_deviation_sigma, presynaptic_group_is_input, random_state_manager->d_states);
			CudaCheckError();

			CudaSafeCall(cudaMemcpy(&presynaptic_neuron_indices[original_number_of_synapses], d_temp_presynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses, cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy(&postsynaptic_neuron_indices[original_number_of_synapses], d_temp_postsynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses, cudaMemcpyDeviceToHost));

			break;
		}

		case CONNECTIVITY_TYPE_GAUSSIAN: // 1-D or 2-D
		{

			float sigma = parameter;

			// For gaussian connectivity, the shape of the layers matters.
			// If we desire a given number of neurons, we must scale the gaussian
			float gaussian_scaling_factor = 1.0f;
			if (parameter_two != 0.0f){
				gaussian_scaling_factor = 0.0f;
				int pre_x = presynaptic_group_shape[0] / 2;
				int pre_y = presynaptic_group_shape[1] / 2;
				for (int i = 0; i < postsynaptic_group_shape[0]; i++){
					for (int j = 0; j < postsynaptic_group_shape[1]; j++){
						// Post XY
						int post_x = i;
						int post_y = j;
						// Distance
						float distance = pow((pow((float)(pre_x - post_x), 2.0f) + pow((float)(pre_y - post_y), 2.0f)), 0.5f);
						// Gaussian Probability
						gaussian_scaling_factor += GAUS(distance, sigma);
					}
				}
				// Multiplying the gaussian scaling factor by the number of synapses you require:
				gaussian_scaling_factor = gaussian_scaling_factor / parameter_two;
			}

			int threads = 512;
			dim3 threads_per_block = dim3(threads);
			int total_number_of_neuron_pairs = (preend - prestart) * (postend - poststart);
			int total_pre_neurons = preend - prestart;
			int total_post_neurons = postend - poststart;
			dim3 neuron_pair_block_dimensions = dim3((total_pre_neurons + threads)/threads, (total_post_neurons + threads)/threads);

			// bool * d_yes_no_connection_matrix;
			// CudaSafeCall(cudaMalloc((void **)&d_yes_no_connection_matrix, sizeof(int)*total_number_of_neuron_pairs));

			for (int k = 0; k < synapse_params->max_number_of_connections_per_pair; k++){

				thrust::device_vector<bool> d_yes_no_connection_vector(total_number_of_neuron_pairs);
				bool * d_yes_no_connection_vector_pointer = thrust::raw_pointer_cast(&d_yes_no_connection_vector[0]);

				compute_yes_no_connection_matrix_for_groups<<<neuron_pair_block_dimensions, threads_per_block>>>(d_yes_no_connection_vector_pointer, presynaptic_group_shape[0], postsynaptic_group_shape[0], postsynaptic_group_shape[1], sigma, total_pre_neurons, total_post_neurons);
				CudaCheckError();
				
				int total_number_of_new_synapses = thrust::count(d_yes_no_connection_vector.begin(), d_yes_no_connection_vector.end(), true);
				// printf("total_number_of_new_synapses: %d\n", total_number_of_new_synapses);

				this->increment_number_of_synapses(total_number_of_new_synapses);

				bool * yes_no_connection_matrix = (bool *)malloc(total_number_of_neuron_pairs*sizeof(bool));
				CudaSafeCall(cudaMemcpy(yes_no_connection_matrix, d_yes_no_connection_vector_pointer, sizeof(bool)*total_number_of_neuron_pairs, cudaMemcpyDeviceToHost));

				int pre_width = presynaptic_group_shape[0];
				int post_width = postsynaptic_group_shape[0];
				int post_height = postsynaptic_group_shape[1];

				int true_count = 0;
				for (int i = prestart; i < preend; i++){
					for (int j = poststart; j < postend; j++){

						int zeroed_pre = i - prestart;
						int zeroed_post = j - poststart;

						int pre_x = zeroed_pre % pre_width;
						int pre_y = floor((float)(zeroed_pre) / pre_width);
						// Post XY
						int post_x = zeroed_post % post_width;
						int post_y = floor((float)(zeroed_post) / post_width);

						int vector_index = pre_x + (pre_width * pre_y) + (post_y * total_pre_neurons) + (post_x * post_height * total_pre_neurons);

						if (yes_no_connection_matrix[vector_index] == true) {
							// Setup Synapses
							presynaptic_neuron_indices[total_number_of_synapses - total_number_of_new_synapses - 1 + true_count] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
							postsynaptic_neuron_indices[total_number_of_synapses - total_number_of_new_synapses - 1 + true_count] = j;

							true_count++;
						}
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
						
						this->increment_number_of_synapses(1);

						// Setup the synapses:
						// Setup Synapses
						presynaptic_neuron_indices[total_number_of_synapses - 1] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
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
			this->increment_number_of_synapses(1);

			// Setup Synapses
			presynaptic_neuron_indices[original_number_of_synapses] = CORRECTED_PRESYNAPTIC_ID(prestart + int(parameter), presynaptic_group_is_input);
			postsynaptic_neuron_indices[original_number_of_synapses] = poststart + int(parameter_two);

			break;
		}
		default:
		{
			print_message_and_exit("Unknown Connection Type.");
			break;
		}
	}

	temp_number_of_synapses_in_last_group = total_number_of_synapses - original_number_of_synapses;

	if (print_synapse_group_details == true) printf("%d new synapses added.\n\n", temp_number_of_synapses_in_last_group);

	for (int i = original_number_of_synapses; i < total_number_of_synapses; i++){
		// printf("i: %d\n", i);
		// Setup Weights
		if (weight_range[0] == weight_range[1]) {
			synaptic_efficacies_or_weights[i] = weight_range[0];
		} else {
			float rndweight = weight_range[0] + (weight_range[1] - weight_range[0])*((float)rand() / (RAND_MAX));
			synaptic_efficacies_or_weights[i] = rndweight;
		}

		original_synapse_indices[i] = i;

	}

}

void Synapses::increment_number_of_synapses(int increment) {

	total_number_of_synapses += increment;

	presynaptic_neuron_indices = (int*)realloc(presynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    postsynaptic_neuron_indices = (int*)realloc(postsynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    synaptic_efficacies_or_weights = (float*)realloc(synaptic_efficacies_or_weights, total_number_of_synapses * sizeof(float));
    // CudaSafeCall(cudaHostAlloc((void**)&synaptic_efficacies_or_weights, total_number_of_synapses * sizeof(float), cudaHostAllocDefault));
    original_synapse_indices = (int*)realloc(original_synapse_indices, total_number_of_synapses * sizeof(int));
    
}


void Synapses::allocate_device_pointers() {

	printf("Allocating synapse device pointers...\n");

	CudaSafeCall(cudaMalloc((void **)&d_presynaptic_neuron_indices, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_postsynaptic_neuron_indices, sizeof(int)*total_number_of_synapses));
	CudaSafeCall(cudaMalloc((void **)&d_synaptic_efficacies_or_weights, sizeof(float)*total_number_of_synapses));

	CudaSafeCall(cudaMemcpy(d_presynaptic_neuron_indices, presynaptic_neuron_indices, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_postsynaptic_neuron_indices, postsynaptic_neuron_indices, sizeof(int)*total_number_of_synapses, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_synaptic_efficacies_or_weights, synaptic_efficacies_or_weights, sizeof(float)*total_number_of_synapses, cudaMemcpyHostToDevice));
}

// Provides order of magnitude speedup for LIF (All to all atleast). 
// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causing massive slowdown.
// Randomising order of synapses means that each block is accessing a larger number of points in memory.
void Synapses::shuffle_synapses() {

	printf("Shuffling synapses...\n");

	std::random_shuffle(&original_synapse_indices[0], &original_synapse_indices[total_number_of_synapses]);

	int* new_presynaptic_neuron_indices = (int *)malloc(total_number_of_synapses*sizeof(int));
	int* new_postsynaptic_neuron_indices = (int *)malloc(total_number_of_synapses*sizeof(int));
	float* new_synaptic_efficacies_or_weights = (float *)malloc(total_number_of_synapses*sizeof(float));
	
	for(int i = 0; i < total_number_of_synapses; i++) {

		// printf("i: %d, postsynaptic_neuron_indices[i]: %d\n", i, postsynaptic_neuron_indices[i]);
		// printf("i: %d, original_synapse_indices[i]: %d\n", i, original_synapse_indices[i]);

		new_presynaptic_neuron_indices[i] = presynaptic_neuron_indices[original_synapse_indices[i]];
		new_postsynaptic_neuron_indices[i] = postsynaptic_neuron_indices[original_synapse_indices[i]];
		new_synaptic_efficacies_or_weights[i] = synaptic_efficacies_or_weights[original_synapse_indices[i]];

	}

	presynaptic_neuron_indices = new_presynaptic_neuron_indices;
	postsynaptic_neuron_indices = new_postsynaptic_neuron_indices;
	synaptic_efficacies_or_weights = new_synaptic_efficacies_or_weights;

}




void Synapses::set_threads_per_block_and_blocks_per_grid(int threads) {

	threads_per_block.x = threads;
	number_of_synapse_blocks_per_grid = dim3(1000);

}


__global__ void compute_yes_no_connection_matrix_for_groups(bool * d_yes_no_connection_vector, int pre_width, int post_width, int post_height, float sigma, int total_pre_neurons, int total_post_neurons) {
	
	int idx_pre = threadIdx.x + blockIdx.x * blockDim.x;
	int idx_post = threadIdx.y + blockIdx.y * blockDim.y;

	if ((idx_pre < total_pre_neurons) && (idx_post < total_post_neurons)) {

		unsigned int seed = 9;
		curandState_t state;
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx_pre * total_pre_neurons + idx_post, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&state);

		float prob = curand_uniform(&state);

		// Get the relative distance from the two neurons
		// Pre XY
		int pre_x = idx_pre % pre_width;
		int pre_y = floor((float)(idx_pre) / pre_width);
		// Post XY
		int post_x = idx_post % post_width;
		int post_y = floor((float)(idx_post) / post_width);

		float distance = norm3df((float)(pre_x - post_x), (float)(pre_y - post_y), 0);

		float gaussian_value = (1.0f/(sigma*(sqrtf(2.0f*M_PI)))) * (expf(-1.0f * (powf((distance),(2.0f))) / (2.0f*(powf(sigma,(2.0f))))));

		int vector_index = pre_x + (pre_width * pre_y) + (post_y * total_pre_neurons) + (post_x * post_height * total_pre_neurons);
		
		d_yes_no_connection_vector[vector_index] = (prob < gaussian_value) ? false : true;

	}
}


__global__ void set_neuron_indices_by_sampling_from_normal_distribution(int total_number_of_new_synapses, int postsynaptic_group_id, int poststart, int prestart, int post_width, int post_height, int pre_width, int pre_height, int number_of_new_synapses_per_postsynaptic_neuron, int number_of_postsynaptic_neurons_in_group, int * d_presynaptic_neuron_indices, int * d_postsynaptic_neuron_indices, float * d_synaptic_efficacies_or_weights, float standard_deviation_sigma, bool presynaptic_group_is_input, curandState_t* d_states) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int t_idx = idx;
	while (idx < total_number_of_new_synapses) {
		
		int postsynaptic_neuron_id = idx / number_of_new_synapses_per_postsynaptic_neuron;
		d_postsynaptic_neuron_indices[idx] = poststart + postsynaptic_neuron_id;

		int postsynaptic_x = postsynaptic_neuron_id % post_width; 
		int postsynaptic_y = floor((float)(postsynaptic_neuron_id) / post_width);
		int fractional_x = postsynaptic_x / post_width;
		int fractional_y = postsynaptic_y / post_height;

		int corresponding_presynaptic_centre_x = pre_width * fractional_x; 
		int corresponding_presynaptic_centre_y = pre_height * fractional_y;

		bool presynaptic_x_set = false;
		bool presynaptic_y_set = false;
		int presynaptic_x = -1;
		int presynaptic_y = -1; 

		while (true) {

			if (presynaptic_x_set == false) {
				float value_from_normal_distribution_for_x = curand_normal(&d_states[t_idx]);
				float scaled_value_from_normal_distribution_for_x = standard_deviation_sigma * value_from_normal_distribution_for_x;
				int rounded_scaled_value_from_normal_distribution_for_x = round(scaled_value_from_normal_distribution_for_x);
				presynaptic_x = corresponding_presynaptic_centre_x + rounded_scaled_value_from_normal_distribution_for_x;
				if ((presynaptic_x > -1) && (presynaptic_x < pre_width)) {
					presynaptic_x_set = true;
				}

			}

			if (presynaptic_y_set == false) {
			
				float value_from_normal_distribution_for_y = curand_normal(&d_states[t_idx]);
				float scaled_value_from_normal_distribution_for_y = standard_deviation_sigma * value_from_normal_distribution_for_y;
				int rounded_scaled_value_from_normal_distribution_for_y = round(scaled_value_from_normal_distribution_for_y);
				presynaptic_y = corresponding_presynaptic_centre_y + rounded_scaled_value_from_normal_distribution_for_y;
				if ((presynaptic_y > -1) && (presynaptic_y < pre_width)) {
					presynaptic_y_set = true;
				}

			}

			if (presynaptic_x_set && presynaptic_y_set) {
				d_presynaptic_neuron_indices[idx] = CORRECTED_PRESYNAPTIC_ID(prestart + presynaptic_x + presynaptic_y*pre_width, presynaptic_group_is_input);
				break;
			}
			

		}	

		idx += blockDim.x * gridDim.x;

	}	

	__syncthreads();

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