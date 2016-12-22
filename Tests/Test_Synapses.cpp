#include "catch.hpp"

/**
		SYNAPSES Test Set
**/
#include "Spike/Synapses/Synapses.hpp"
#include "Spike/Neurons/Neurons.hpp"


TEST_CASE("Synapses Class Tests") {
	Synapses test_synapses;
	Neurons test_neurons;
	float timestep = 0.1f;

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
	int dim2_2 = 250;

	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);


	SECTION("Constructor Beginning with no synapses"){
		REQUIRE(test_synapses.total_number_of_synapses == 0);
	}

	SECTION("AddGroup All to All"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				REQUIRE(test_synapses.presynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == i);
				REQUIRE(test_synapses.postsynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == j + test_neurons.group_shapes[presynaptic_population][1]);
			}
		}
	}

	SECTION("AddGroup One to One"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
		test_synapses.AddGroup(
			presynaptic_population,
			presynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			REQUIRE(test_synapses.presynaptic_neuron_indices[i] == i);
			REQUIRE(test_synapses.postsynaptic_neuron_indices[i] == i);
		}
	}

	SECTION("AddGroup Random"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_RANDOM;
		synapse_params.random_connectivity_probability = 0.5f;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		REQUIRE(std::abs(synapse_params.random_connectivity_probability - ((float)test_synapses.total_number_of_synapses / (float)(dim2*dim2_2))) < 0.05);
	}

	SECTION("AddGroup Gaussian Sample"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		synapse_params.gaussian_synapses_per_postsynaptic_neuron = 25;
		synapse_params.gaussian_synapses_standard_deviation = 5;
		
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		int average_standard_dev = 0;
		for(int i=0; i<dim2_2; i++){
			int num_connects = 0;
			int presynaptic_mean_expected = ((float)i / dim2_2)*dim2;
			int presynaptic_mean_actual = 0;
			int standard_deviation = 0;
			for(int j=0; j<test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == 100 + i){
					num_connects++;
					presynaptic_mean_actual += test_synapses.presynaptic_neuron_indices[j];
					standard_deviation += pow((test_synapses.presynaptic_neuron_indices[j] - presynaptic_mean_expected), 2);
				}
			}
			// Check that the number of connections are correct
			REQUIRE(num_connects == synapse_params.gaussian_synapses_per_postsynaptic_neuron);
			// Check that the mean is within 10 neurons of what it should be
			REQUIRE(std::abs(presynaptic_mean_expected - presynaptic_mean_actual/num_connects) < 10);
			
			average_standard_dev += pow(((float)standard_deviation  / (num_connects - 1)), 0.5);
		}
		average_standard_dev /= dim2_2;
		REQUIRE(std::abs(average_standard_dev - synapse_params.gaussian_synapses_standard_deviation) < 0.25*synapse_params.gaussian_synapses_standard_deviation);

	}

	// SECTION("AddGroup Single Connectivity"){
	// 	synapse_parameters_struct synapse_params;
	// 	synapse_params.connectivity_type = CONNECTIVITY_TYPE_SINGLE;
	// 	synapse_params.pairwise_connect_presynaptic = 25;
	// 	synapse_params.pairwise_connect_postsynaptic = 10;
		
	// 	test_synapses.AddGroup(
	// 		presynaptic_population,
	// 		postsynaptic_population,
	// 		&test_neurons,
	// 		&test_neurons,
	// 		timestep,
	// 		&synapse_params);

	// 	REQUIRE(test_synapses.presynaptic_neuron_indices[0] == 25);
	// 	REQUIRE(test_synapses.postsynaptic_neuron_indices[0] == 10 + 100);
	// }

	SECTION("Constant Weight Range"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		synapse_params.weight_range_bottom = 2.7f;
		synapse_params.weight_range_top = 2.7f;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == synapse_params.weight_range_bottom);
			}
		}
	}
	SECTION("Uniform Weight Range"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		synapse_params.weight_range_bottom = 0.0f;
		synapse_params.weight_range_top = 10.0f;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		float mean_weight = 0.0f;
		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				mean_weight += test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]];
				REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] >= synapse_params.weight_range_bottom);
				REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] <= synapse_params.weight_range_top);
			}
		}

		mean_weight /= test_neurons.group_shapes[presynaptic_population][1]*test_neurons.group_shapes[postsynaptic_population][1];
		// Check the mean is within 10% of what it should be
		REQUIRE(std::abs(mean_weight - synapse_params.weight_range_top / 2.0f) < 0.01*synapse_params.weight_range_top);
	}

	SECTION("Input Neuron Indices Check"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		
		test_synapses.AddGroup(
			-1,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				REQUIRE(test_synapses.presynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == - 1 - i);
				REQUIRE(test_synapses.postsynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == j + test_neurons.group_shapes[presynaptic_population][1]);
			}
		}
	}

	SECTION("Synapse Shuffle Check"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
		test_synapses.AddGroup(
			presynaptic_population,
			presynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Shuffle the created connections
		test_synapses.shuffle_synapses();
		
		// Create an array to tick off synapses
		bool* check_array = (bool*)malloc(sizeof(bool)*dim2);
		for (int i=0; i < dim2; i++){
			check_array[i] = false;
		}

		// Check the synapses
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			REQUIRE(test_synapses.presynaptic_neuron_indices[i] == test_synapses.postsynaptic_neuron_indices[i]);
			if (check_array[test_synapses.presynaptic_neuron_indices[i]]){
				REQUIRE(false == true);
			} else {
				check_array[test_synapses.presynaptic_neuron_indices[i]] = true;
			}
		}

	}
}


/**
		SPIKINGSYNAPSES Test Set
**/
#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
TEST_CASE("Spiking Synapses Class Tests") {
	SpikingSynapses test_synapses;
	SpikingNeurons test_neurons;

	SECTION("Constructor"){
		REQUIRE(test_synapses.maximum_axonal_delay_in_timesteps == 0);
	}

	float timestep = 0.1f;

	// Creating Neuron Populations to test with
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
	int dim2_2 = 250;

	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

	SECTION("Delay/STDP Setting Check"){
		SECTION("Constant Delay Value"){
			spiking_synapse_parameters_struct synapse_params;
			synapse_params.stdp_on = false;
			synapse_params.delay_range[0] = 0.1f;
			synapse_params.delay_range[1] = 0.1f;
			synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
			test_synapses.AddGroup(
				presynaptic_population,
				postsynaptic_population,
				&test_neurons,
				&test_neurons,
				timestep,
				&synapse_params);
			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				REQUIRE(test_synapses.delays[i] == (int)(synapse_params.delay_range[0] / timestep));
				REQUIRE(test_synapses.stdp[i] == false);
			}
			REQUIRE(test_synapses.maximum_axonal_delay_in_timesteps == 1);
		}

		SECTION("Uniform Delay (+STDP) Value"){
			spiking_synapse_parameters_struct synapse_params;
			synapse_params.stdp_on = true;
			synapse_params.delay_range[0] = 0.1f;
			synapse_params.delay_range[1] = 5.0f;
			synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
			test_synapses.AddGroup(
				presynaptic_population,
				postsynaptic_population,
				&test_neurons,
				&test_neurons,
				timestep,
				&synapse_params);
			float mean_delay = 0;
			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				REQUIRE(test_synapses.delays[i] >= (int)(synapse_params.delay_range[0]) / timestep);
				REQUIRE(test_synapses.delays[i] <= (int)(synapse_params.delay_range[1]) / timestep);
				REQUIRE(test_synapses.stdp[i] == true);
				mean_delay += test_synapses.delays[i];
			}
			// Ensure that the mean delay is within a timestep of what it should be
			REQUIRE(std::abs(((float)mean_delay / test_synapses.total_number_of_synapses) 
				- ((synapse_params.delay_range[0] + synapse_params.delay_range[1]) / (2 * timestep))) < 1.0f);
			REQUIRE(test_synapses.maximum_axonal_delay_in_timesteps == 50);
		}
	}


	SECTION("CUDA Kernels for Spikes approaching Synapses"){
		spiking_synapse_parameters_struct synapse_params;
		synapse_params.stdp_on = true;
		synapse_params.delay_range[0] = 0.1f;
		synapse_params.delay_range[1] = 5.0f;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Allocate Neuron Pointers
		test_neurons.set_threads_per_block_and_blocks_per_grid(512);
		test_synapses.set_threads_per_block_and_blocks_per_grid(512);

		SECTION("Default Device Parameter States") {
			test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, false);
			test_synapses.allocate_device_pointers();
			// Set-up Variables
			test_neurons.copy_constants_to_device();
			test_synapses.copy_constants_and_initial_efficacies_to_device();

			test_neurons.reset_neuron_activities();
			test_synapses.reset_synapse_activities();
			// Run the synapse interaction
			float current_time = 0.9f;
			test_synapses.interact_spikes_with_synapses(&test_neurons, &test_neurons, current_time, timestep);
			// Copy back device variables and check
			float* last_synapse_spike_times;
			int* timestep_countdown;
			last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
			timestep_countdown = (int*)malloc(sizeof(int)*test_synapses.total_number_of_synapses);
			CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy(timestep_countdown, test_synapses.d_spikes_travelling_to_synapse, sizeof(int)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));

			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				REQUIRE(last_synapse_spike_times[i] == -1000.0f);
				REQUIRE(timestep_countdown[i] == -1);
			}
		}

		SECTION("Low Fidelity Spike Recognition") {
			test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, false);
			test_synapses.allocate_device_pointers();

			// Set-up Variables
			test_neurons.copy_constants_to_device();
			test_synapses.copy_constants_and_initial_efficacies_to_device();

			test_neurons.reset_neuron_activities();
			test_synapses.reset_synapse_activities();
			float* last_neuron_spike_times;
			last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
			// Set of the neuron spike times to now
			float current_time = 0.9f;
			int indices[5] = {0, 12, 78, 9, 11};
			for (int i=0; i < 5; i++){
				last_neuron_spike_times[indices[i]] = current_time;
			}
			// Return the data to the device
			CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_neuron_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));
			// Run the sim
			float* last_synapse_spike_times;
			int* timestep_countdown;
			test_synapses.interact_spikes_with_synapses(&test_neurons, &test_neurons, current_time, timestep);
			last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
			timestep_countdown = (int*)malloc(sizeof(int)*test_synapses.total_number_of_synapses);
			CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy(timestep_countdown, test_synapses.d_spikes_travelling_to_synapse, sizeof(int)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));

			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				if ((test_synapses.presynaptic_neuron_indices[i] == indices[0]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[1]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[2]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[3]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[4])){
					REQUIRE(last_synapse_spike_times[i] == -1000.0f);
					REQUIRE(timestep_countdown[i] == test_synapses.delays[i]);
				} else {
					REQUIRE(last_synapse_spike_times[i] == -1000.0f);
					REQUIRE(timestep_countdown[i] == -1);
				}
			}
		}

		SECTION("Low Fidelity Time Allocation Check"){
			test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, false);
			test_synapses.allocate_device_pointers();
			// Set-up Variables
			test_neurons.copy_constants_to_device();
			test_synapses.copy_constants_and_initial_efficacies_to_device();
			test_neurons.reset_neuron_activities();
			test_synapses.reset_synapse_activities();
			int* spike_countdown;
			spike_countdown = (int*)malloc(sizeof(int)*test_synapses.total_number_of_synapses);
			CudaSafeCall(cudaMemcpy(spike_countdown, test_synapses.d_spikes_travelling_to_synapse, sizeof(int)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
			// Set of the neuron spike times to now
			float current_time = 0.9f;
			int indices[5] = {0, 12, 78, 9, 11};
			for (int i=0; i < 5; i++){
				spike_countdown[indices[i]] = 1;
			}
			// Return the data to the device
			CudaSafeCall(cudaMemcpy(test_synapses.d_spikes_travelling_to_synapse, spike_countdown, sizeof(int)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));
			// Run the sim
			test_synapses.interact_spikes_with_synapses(&test_neurons, &test_neurons, current_time, timestep);
			float* last_synapse_spike_times;
			last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
			CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));

			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				if ((i == indices[0]) ||
						(i == indices[1]) ||
						(i == indices[2]) ||
						(i == indices[3]) ||
						(i == indices[4])){
					REQUIRE(last_synapse_spike_times[i] == current_time);
				} else {
					REQUIRE(last_synapse_spike_times[i] == -1000.0f);
				}
			}
		}

		SECTION("High Fidelity Spike Time Allocation Check") {
			test_neurons.allocate_device_pointers(test_synapses.maximum_axonal_delay_in_timesteps, true);
			test_synapses.allocate_device_pointers();
			// Set-up Variables
			test_neurons.copy_constants_to_device();
			test_synapses.copy_constants_and_initial_efficacies_to_device();

			test_neurons.reset_neuron_activities();
			test_synapses.reset_synapse_activities();

			// Copy the correct array
			char* neuron_spike_array;
			neuron_spike_array = (char*)malloc(sizeof(char)*test_neurons.total_number_of_neurons*((test_synapses.maximum_axonal_delay_in_timesteps / 8)+ 1));
			CudaSafeCall(cudaMemcpy(neuron_spike_array, test_neurons.d_bitarray_of_neuron_spikes, sizeof(char)*test_neurons.total_number_of_neurons*((test_synapses.maximum_axonal_delay_in_timesteps / 8)+ 1), cudaMemcpyDeviceToHost));
			// Set some indices to have spiked at t=2.6s
			float current_time = 0.0f;
			int indices[5] = {0, 12, 78, 9, 11};
			// Check the neuron spikes array
			for (int n=0; n < 5; n++){
				for (int j=0; j < 8; j++){
					char byte = neuron_spike_array[indices[n]*((test_synapses.maximum_axonal_delay_in_timesteps / 8)+ 1) + ((test_synapses.maximum_axonal_delay_in_timesteps - 26) / 8)];
					byte |= 1 << ((test_synapses.maximum_axonal_delay_in_timesteps - 26) % 8);
					neuron_spike_array[indices[n]*((test_synapses.maximum_axonal_delay_in_timesteps / 8)+ 1) + ((test_synapses.maximum_axonal_delay_in_timesteps - 26) / 8)] = byte;
				}
			}

			// Copy back the array
			CudaSafeCall(cudaMemcpy(test_neurons.d_bitarray_of_neuron_spikes, neuron_spike_array, sizeof(char)*test_neurons.total_number_of_neurons*((test_synapses.maximum_axonal_delay_in_timesteps / 8)+ 1), cudaMemcpyHostToDevice));
			
			// Run the sim
			test_synapses.interact_spikes_with_synapses(&test_neurons, &test_neurons, current_time, timestep);
			
			float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
			CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));

			for (int i=0; i < test_synapses.total_number_of_synapses; i++){
				if ((test_synapses.presynaptic_neuron_indices[i] == indices[0]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[1]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[2]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[3]) ||
						(test_synapses.presynaptic_neuron_indices[i] == indices[4])){
					if (test_synapses.delays[i] == 26){
						REQUIRE(last_synapse_spike_times[i] == current_time);
					} else {
						REQUIRE(last_synapse_spike_times[i] == -1000.0f);
					}
				} else {
					REQUIRE(last_synapse_spike_times[i] == -1000.0f);
				}
			}
		}

	}
}


/**
		CURRENTSPIKINGSYNAPSES Test Set
**/
#include "Spike/Synapses/CurrentSpikingSynapses.hpp"
TEST_CASE("Current Spiking Synapses Class Tests") {
	CurrentSpikingSynapses test_synapses;
	SpikingNeurons test_neurons;

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
	int dim2_2 = 250;
	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

	// Setting times
	float timestep = 0.1f;
	float current_time = 0.9f;

	// Setting Synapses up!
	spiking_synapse_parameters_struct synapse_params;
	synapse_params.stdp_on = true;
	synapse_params.delay_range[0] = 0.1f;
	synapse_params.delay_range[1] = 5.0f;
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
	// Set-up Variables
	test_neurons.copy_constants_to_device();
	test_synapses.copy_constants_and_initial_efficacies_to_device();
	test_neurons.reset_neuron_activities();
	test_synapses.reset_synapse_activities();
	// Check the initial current injection values after a run
	test_synapses.calculate_postsynaptic_current_injection(&test_neurons, current_time, timestep);
	float* current_inj = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
	CudaSafeCall(cudaMemcpy(current_inj, test_neurons.d_current_injections, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
	for (int i=0; i < test_neurons.total_number_of_neurons; i++){
		REQUIRE(current_inj[i] == 0.0f);
	}
	
	// Set some of the synapses as having a spike at them
	float* last_synapse_spike_times = (float*)malloc(sizeof(float)*test_synapses.total_number_of_synapses);
	CudaSafeCall(cudaMemcpy(last_synapse_spike_times, test_synapses.d_time_of_last_spike_to_reach_synapse, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyDeviceToHost));
	int indices[5] = {0, 120, 78, 9, 11};
	for (int i=0; i < 5; i++){
		for (int j=0; j < test_synapses.total_number_of_synapses; j++){
			if (test_synapses.postsynaptic_neuron_indices[j] == indices[i]){
				last_synapse_spike_times[j] = current_time;
			}
		}
	}
	CudaSafeCall(cudaMemcpy(test_synapses.d_time_of_last_spike_to_reach_synapse, last_synapse_spike_times, sizeof(float)*test_synapses.total_number_of_synapses, cudaMemcpyHostToDevice));
	// Now running the sim and checking again
	test_synapses.calculate_postsynaptic_current_injection(&test_neurons, current_time, timestep);
	CudaSafeCall(cudaMemcpy(current_inj, test_neurons.d_current_injections, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
	for(int i=0; i < test_neurons.total_number_of_neurons; i++){
		float current = 0.0f;
		if ((i == indices[0]) || (i == indices[1]) || (i == indices[2]) || (i == indices[3]) || (i == indices[4])){
			for (int j=0; j < test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == i){
					current += test_synapses.synaptic_efficacies_or_weights[j];
				}
			}
		}
		REQUIRE(std::abs(current_inj[i] - current) < 0.00005f);
	}
}


/**
		CONDUCTANCESPIKINGSYNAPSES Test Set
**/
#include "Spike/Synapses/ConductanceSpikingSynapses.hpp"
TEST_CASE("Conductance Spiking Synapses Class Tests") {
  // TODO
}

