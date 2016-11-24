#include "catch.hpp"
#include "../Helpers/CUDAErrorCheckHelpers.h"

/**
		HIGGINSSTDP.CU Test Set
**/
#include "../STDP/STDP.h"
#include "../STDP/HigginsSTDP.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"
TEST_CASE("HigginsSTDP") {
	SpikingSynapses test_synapses;
	SpikingNeurons test_neurons;
	SpikingNeurons input_neurons;

	// Assigning these neurons and synapses to the
	HigginsSTDP test_stdp;
	higgins_stdp_parameters_struct * STDP_PARAMS = new higgins_stdp_parameters_struct();
	test_stdp.Set_STDP_Parameters(
		(SpikingSynapses*) &test_synapses,
		(SpikingNeurons*) &test_neurons,
		(SpikingNeurons*) &input_neurons,
		STDP_PARAMS);

	// Pre-synaptic Population
	neuron_parameters_struct neuron_params_1;
	int dim1 = 1;
	int dim2 = 100;
	neuron_params_1.group_shape[0] = dim1;
	neuron_params_1.group_shape[1] = dim2;

	int presynaptic_population = test_neurons.AddGroup(&neuron_params_1);

	// Post-synaptic Population
	neuron_parameters_struct neuron_params_2;
	int dim1_2 = 1;
	int dim2_2 = 100;
	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

	// Setting times
	float timestep = 0.01f;
	float current_time = 0.09f;

	// Setting Synapses up!
	spiking_synapse_parameters_struct synapse_params;
	synapse_params.stdp_on = true;
	synapse_params.delay_range[0] = 0.1f;
	synapse_params.delay_range[1] = 0.1f;
	// Set a fixed weight range
	synapse_params.weight_range_bottom = 0.5f;
	synapse_params.weight_range_top = 0.5f;
	// Connect
	synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	test_synapses.AddGroup(
		presynaptic_population,
		postsynaptic_population,
		&test_neurons,
		&test_neurons,
		timestep,
		&synapse_params);

	// First ensuring that the weights are all 0.5f:
	SECTION("Testing initial weight values"){
		for(int i=0; i < test_synapses.total_number_of_synapses; i++){
			REQUIRE(test_synapses.synaptic_efficacies_or_weights[i] == 0.5f);
		}
	}

	// Setting up simulation
	test_neurons.set_threads_per_block_and_blocks_per_grid(512);
	test_synapses.set_threads_per_block_and_blocks_per_grid(512);
	// Allocating Pointers
	test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, true);
	test_synapses.allocate_device_pointers();
	test_stdp.allocate_device_pointers();
	// Set-up Variables
	test_neurons.copy_constants_to_device();
	test_synapses.copy_constants_and_initial_efficacies_to_device();
	test_neurons.reset_neuron_activities();
	test_synapses.reset_synapse_activities();
	test_stdp.reset_STDP_activities();

	SECTION("LTP Test"){
		// Set some of the synapses as having a spike at them
		float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		int indices[5] = {0, 12, 78, 9, 11};
		float spike_times[5] = {0.01f, 0.02f, 0.05f, 0.08f, 0.00f};
		for (int i=0; i < 5; i++){
			for (int j=0; j < test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == indices[i]){
					last_synapse_spike_times[j] = spike_times[int(j % 5)];
				}
			}
		}
		CudaSafeCall(cudaMemcpy(test_synapses.d_time_of_last_spike_to_reach_synapse, last_synapse_spike_times, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));


		// Set neuron last spike indices to those required:
		float* last_neuron_spike_times;
		last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Set of the neuron spike times to now
		for (int i=0; i < 5; i++){
			// Setting it to half the current_time so that it can
			last_neuron_spike_times[indices[i]] = current_time;
		}
		// Return the data to the device
		CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));

		// Now run STDP
		test_stdp.Run_STDP(test_neurons.d_last_spike_time_of_each_neuron, current_time, timestep);

		// Check synaptic weights copy back weights
		float* synaptic_weights = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(synaptic_weights, test_synapses.d_synaptic_efficacies_or_weights, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		// Check the synaptic weight values
		for(int i=0; i < test_synapses.total_number_of_synapses; i++){
			float diff = current_time - last_synapse_spike_times[i];
			float weight_change = (STDP_PARAMS->w_max - 0.5f) * (STDP_PARAMS->a_plus * exp(-diff / STDP_PARAMS->tau_plus));
			REQUIRE(synaptic_weights[i] >= 0.5f);
			REQUIRE(std::abs(synaptic_weights[i] - (0.5f + weight_change)) < 0.00005f);
		}
	}

	SECTION("LTD Test"){
		// Set some of the synapses as having a spike at them
		float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		int indices[5] = {0, 12, 78, 9, 11};
		float spike_times[5] = {0.01f, 0.02f, 0.05f, 0.08f, 0.00f};
		for (int i=0; i < 5; i++){
			for (int j=0; j < test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == indices[i]){
					last_synapse_spike_times[j] = current_time;
				}
			}
		}
		CudaSafeCall(cudaMemcpy(test_synapses.d_time_of_last_spike_to_reach_synapse, last_synapse_spike_times, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));


		// Set neuron last spike indices to those required:
		float* last_neuron_spike_times;
		last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Set of the neuron spike times to now
		for (int i=0; i < 5; i++){
			// Setting it to half the current_time so that it can
			last_neuron_spike_times[indices[i]] = spike_times[i];
		}
		// Return the data to the device
		CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));

		// Now run STDP
		test_stdp.Run_STDP(test_neurons.d_last_spike_time_of_each_neuron, current_time, timestep);

		// Check synaptic weights copy back weights
		float* synaptic_weights = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(synaptic_weights, test_synapses.d_synaptic_efficacies_or_weights, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		// Check the synaptic weight values
		for(int i=0; i < test_synapses.total_number_of_synapses; i++){
			float diff = last_neuron_spike_times[test_synapses.postsynaptic_neuron_indices[i]] - current_time;
			float weightscale = STDP_PARAMS->w_max * STDP_PARAMS->a_minus * exp(diff / STDP_PARAMS->tau_minus);
			REQUIRE(synaptic_weights[i] <= 0.5f);
			REQUIRE(std::abs(synaptic_weights[i] - (0.5f + weightscale)) < 0.00005f);
		}
	}
}

/**
		EVANSSTDP.CU Test Set
**/
#include "../STDP/EvansSTDP.h"
TEST_CASE("EvansSTDP") {

}

/**
		MASQUELIERSTDP.CU Test Set
**/
#include "../STDP/MasquelierSTDP.h"
TEST_CASE("MasquelierSTDP") {
	SpikingSynapses test_synapses;
	SpikingNeurons test_neurons;
	SpikingNeurons input_neurons;

	// Assigning these neurons and synapses to the
	MasquelierSTDP test_stdp;
	masquelier_stdp_parameters_struct* STDP_PARAMS = new masquelier_stdp_parameters_struct();
	test_stdp.Set_STDP_Parameters(
		(SpikingSynapses*) &test_synapses,
		(SpikingNeurons*) &test_neurons,
		(SpikingNeurons*) &input_neurons,
		STDP_PARAMS);

	// Pre-synaptic Population
	neuron_parameters_struct neuron_params_1;
	int dim1 = 1;
	int dim2 = 100;
	neuron_params_1.group_shape[0] = dim1;
	neuron_params_1.group_shape[1] = dim2;

	int presynaptic_population = test_neurons.AddGroup(&neuron_params_1);

	// Post-synaptic Population
	neuron_parameters_struct neuron_params_2;
	int dim1_2 = 1;
	int dim2_2 = 100;
	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

	// Setting times
	float timestep = 0.0001f;
	float current_time = 0.09f;

	// Setting Synapses up!
	spiking_synapse_parameters_struct synapse_params;
	synapse_params.stdp_on = true;
	synapse_params.delay_range[0] = 0.1f;
	synapse_params.delay_range[1] = 0.1f;
	// Set a fixed weight range
	synapse_params.weight_range_bottom = 0.5f;
	synapse_params.weight_range_top = 0.5f;
	// Connect
	synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	test_synapses.AddGroup(
		presynaptic_population,
		postsynaptic_population,
		&test_neurons,
		&test_neurons,
		timestep,
		&synapse_params);

	// Setting up simulation
	test_neurons.set_threads_per_block_and_blocks_per_grid(512);
	test_synapses.set_threads_per_block_and_blocks_per_grid(512);
	// Allocating Pointers
	test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, true);
	test_synapses.allocate_device_pointers();
	test_stdp.allocate_device_pointers();
	// Set-up Variables
	test_neurons.copy_constants_to_device();
	test_synapses.copy_constants_and_initial_efficacies_to_device();
	test_neurons.reset_neuron_activities();
	test_synapses.reset_synapse_activities();
	test_stdp.reset_STDP_activities();

	// First ensuring that the weights are all 0.5f:
	SECTION("Testing initial weight values"){
		for(int i=0; i < test_synapses.total_number_of_synapses; i++){
			REQUIRE(test_synapses.synaptic_efficacies_or_weights[i] == 0.5f);
		}
	}

	SECTION("LTP Test"){
		// Set some of the synapses as having a spike at them
		float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		int indices[5] = {110, 111, 177, 108, 114};
		float spike_times[5] = {0.01f, 0.075f, 0.05f, 0.08f, 0.02f};
		for (int i=0; i < 5; i++){
			for (int j=0; j < test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == indices[i]){
					last_synapse_spike_times[j] = spike_times[int(indices[i] % 5)];
				}
			}
		}
		CudaSafeCall(cudaMemcpy(test_synapses.d_time_of_last_spike_to_reach_synapse, last_synapse_spike_times, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));


		// Set neuron last spike indices to those required:
		float* last_neuron_spike_times;
		last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		int* index_of_last_afferent_synapse_to_spike = (int*)malloc(sizeof(int)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		CudaSafeCall(cudaMemcpy(index_of_last_afferent_synapse_to_spike, test_stdp.d_index_of_last_afferent_synapse_to_spike, sizeof(int)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Set of the neuron spike times to now
		for (int i=0; i < 5; i++){
			// Setting it to half the current_time so that it can
			last_neuron_spike_times[indices[i]] = current_time;
		}
		// Find the nearest spike time for each neuron and use it to set index
		for (int j=0; j < test_synapses.total_number_of_synapses; j++){
			// If this synapse has spiked
			if (last_synapse_spike_times[j] >= 0.0f){
				// If the this index is already set
				if (index_of_last_afferent_synapse_to_spike[test_synapses.postsynaptic_neuron_indices[j]] >= 0){
					// Check that the new index is better and set it
					if (last_synapse_spike_times[index_of_last_afferent_synapse_to_spike[test_synapses.postsynaptic_neuron_indices[j]]] < last_synapse_spike_times[j]){
						index_of_last_afferent_synapse_to_spike[test_synapses.postsynaptic_neuron_indices[j]] = j;
					}
				} else {
					index_of_last_afferent_synapse_to_spike[test_synapses.postsynaptic_neuron_indices[j]] = j;
				}
			}
		}
		// Return the data to the device
		CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(test_stdp.d_index_of_last_afferent_synapse_to_spike, index_of_last_afferent_synapse_to_spike, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));

		// Now run STDP
		test_stdp.Run_STDP(test_neurons.d_last_spike_time_of_each_neuron, current_time, timestep);

		// Check synaptic weights copy back weights
		float* synaptic_weights = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(synaptic_weights, test_synapses.d_synaptic_efficacies_or_weights, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		// Check the synaptic weight values
		for(int i=0; i < test_neurons.total_number_of_neurons; i++){
			if (index_of_last_afferent_synapse_to_spike[i] >= 0){
				float diff = current_time - last_synapse_spike_times[index_of_last_afferent_synapse_to_spike[i]];
				// Calculating the weight change
				float weight_change = STDP_PARAMS->a_plus*exp(-diff / STDP_PARAMS->tau_plus);
				// The change should not be applied if time diff is > 7 times the time constant
				if (diff > 7*STDP_PARAMS->tau_plus){
					weight_change = 0.0f;
				}
				// The weight should not increase beyond 1.0
				if (weight_change > 0.5f){
					weight_change = 0.5f;
				}
				// if (last_synapse_spike_times[index_of_last_afferent_synapse_to_spike[i]] > 0.0f){
				// 	printf("Weight: %f, Change: %f, %f\n", synaptic_weights[index_of_last_afferent_synapse_to_spike[i]], weight_change, diff);
				// }
				REQUIRE(std::abs(synaptic_weights[index_of_last_afferent_synapse_to_spike[i]] - (0.5f + weight_change)) < 0.00005f);
			}
		}
	}

	SECTION("LTD Test"){
		// Set some of the synapses as having a spike at them
		float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		int indices[5] = {100, 112, 178, 109, 111};
		float spike_times[5] = {0.01f, 0.02f, 0.05f, 0.08f, 0.089f};
		for (int i=0; i < 5; i++){
			for (int j=0; j < test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == indices[i]){
					last_synapse_spike_times[j] = current_time;
					break;
				}
			}
		}
		CudaSafeCall(cudaMemcpy(test_synapses.d_time_of_last_spike_to_reach_synapse, last_synapse_spike_times, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));


		// Set neuron last spike indices to those required:
		float* last_neuron_spike_times;
		last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Set of the neuron spike times to now
		for (int i=0; i < 5; i++){
			// Setting it to half the current_time so that it can
			last_neuron_spike_times[indices[i]] = spike_times[i];
		}
		// Return the data to the device
		CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));

		// Now run STDP
		test_stdp.Run_STDP(test_neurons.d_last_spike_time_of_each_neuron, current_time, timestep);

		// Check synaptic weights copy back weights
		float* synaptic_weights = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
		CudaSafeCall(cudaMemcpy(synaptic_weights, test_synapses.d_synaptic_efficacies_or_weights, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
		// Check the synaptic weight values
		for(int i=0; i < test_synapses.total_number_of_synapses; i++){
			if (last_synapse_spike_times[i] == current_time){
				float diff = current_time - last_neuron_spike_times[test_synapses.postsynaptic_neuron_indices[i]];
				float weight_change = STDP_PARAMS->a_minus*exp(-diff / STDP_PARAMS->tau_minus);
				if (weight_change > 0.5){
					weight_change = 0.5f;
				}
				// printf("LTD, Weight: %f, Change: %f, %f\n", synaptic_weights[i], weight_change, diff);
				REQUIRE(std::abs(synaptic_weights[i] - (0.5f - weight_change)) < 0.00005f);
			}
		}
	}
}
