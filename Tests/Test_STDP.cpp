#include "catch.hpp"

/**
		HIGGINSSTDP Test Set
**/
#include "Spike/STDP/STDP.hpp"
#include "Spike/STDP/HigginsSTDP.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Synapses/SpikingSynapses.hpp"
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
		EVANSSTDP Test Set
**/
#include "Spike/STDP/EvansSTDP.hpp"
TEST_CASE("EvansSTDP") {
  // TODO
}

/**
		MASQUELIERSTDP Test Set
**/
#include "Spike/STDP/MasquelierSTDP.hpp"
TEST_CASE("MasquelierSTDP") {
  // TODO
}
