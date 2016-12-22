#include "catch.hpp"

/**
		NEURONS Test Set
**/
#include "Spike/Neurons/Neurons.hpp"
TEST_CASE("Neurons Class") {
	// Create an instance of the neuron class
	Neurons test_neurons;

	SECTION("Constructor begins with zero neurons/groups") {
		REQUIRE(test_neurons.total_number_of_neurons == 0);
		REQUIRE(test_neurons.total_number_of_groups == 0);
	}

	SECTION("Adding a Group"){
		neuron_parameters_struct params;
		int dim1 = 1;
		int dim2 = 10;

		params.group_shape[0] = dim1;
		params.group_shape[1] = dim2;

		int ID = test_neurons.AddGroup(&params);

		REQUIRE(ID == 0);
		REQUIRE(test_neurons.total_number_of_neurons == dim1*dim2);
		REQUIRE(test_neurons.start_neuron_indices_for_each_group[0] == 0);
		REQUIRE(test_neurons.last_neuron_indices_for_each_group[0] == (dim1*dim2 - 1));
		REQUIRE(test_neurons.group_shapes[0][0] == dim1);
		REQUIRE(test_neurons.group_shapes[0][1] == dim2);
	}
}



/**
		SPIKINGNEURONS Test Set
**/
#include "Spike/Neurons/SpikingNeurons.hpp"
TEST_CASE("Spiking Neurons Class") {
	// Create an instance of the neuron class
	SpikingNeurons test_neurons;

	spiking_neuron_parameters_struct params;
	int dim1 = 1;
	int dim2 = 10;
	float resting_pot = -50.0f;
	float threshold = 50.0f;

	params.group_shape[0] = dim1;
	params.group_shape[1] = dim2;
	params.resting_potential_v0 = resting_pot;
	params.threshold_for_action_potential_spike = threshold;

	test_neurons.AddGroup(&params);

	SECTION("AddGroup Setting Spiking Parameters") {
		for (int i=0; i < dim1; i++){
			for (int j=0; j < dim2; j ++){
				REQUIRE(test_neurons.after_spike_reset_membrane_potentials_c[j + i*dim2] == resting_pot);
				REQUIRE(test_neurons.thresholds_for_action_potential_spikes[j + i*dim2] == threshold);
			}
		}
	}


	SECTION("Low Fidelity check_for_neuron_spikes Kernel Check") {
		// Setting up the Spiking Neurons
		// No high fidelity
		test_neurons.set_threads_per_block_and_blocks_per_grid(512);
		test_neurons.allocate_device_pointers(0, false);
		// Selecting some indices to set to fire
		int indices[3];
		indices[0] = 1; indices[1] = 3; indices[2] = 5;
		for (int i=0; i < 3; i++){
			test_neurons.after_spike_reset_membrane_potentials_c[indices[i]] = 60.0f;
		}
		// Copying the given arrays to the GPU
		test_neurons.copy_constants_to_device();
		test_neurons.reset_neuron_activities();
		// Carrying out the spike check
		float current_time = 0.9f;
		float timestep = 0.1f;
		test_neurons.check_for_neuron_spikes(current_time, timestep);
		// Copy back the correct array
		float* neuron_spike_times;
		neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Check the neuron spikes array
		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
			if (i == indices[0] || i == indices[1] || i == indices[2]){
				REQUIRE(neuron_spike_times[i] == current_time);
			} else {
				REQUIRE(neuron_spike_times[i] == -1000.0f);
			}
		}
	}


	SECTION("High Fidelity check_for_neuron_spikes Kernel Check") {
		// Setting up the Spiking Neurons
		// No high fidelity
		int max_delay = 10;
		test_neurons.set_threads_per_block_and_blocks_per_grid(512);
		test_neurons.allocate_device_pointers(max_delay, true);
		// Selecting some indices to set to fire
		int indices[3];
		indices[0] = 1; indices[1] = 3; indices[2] = 5;
		for (int i=0; i < 3; i++){
			test_neurons.after_spike_reset_membrane_potentials_c[indices[i]] = 60.0f;
		}
		// Copying the given arrays to the GPU
		test_neurons.copy_constants_to_device();
		test_neurons.reset_neuron_activities();
		// Carrying out the spike check
		float current_time = 0.9f;
		float timestep = 0.1f;
		test_neurons.check_for_neuron_spikes(current_time, timestep);
		// Copy back the correct array
		char* neuron_spike_array;
		neuron_spike_array = (char*)malloc(sizeof(char)*test_neurons.total_number_of_neurons*((max_delay / 8)+ 1));
		CudaSafeCall(cudaMemcpy(neuron_spike_array, test_neurons.d_bitarray_of_neuron_spikes, sizeof(char)*test_neurons.total_number_of_neurons*((max_delay / 8)+ 1), cudaMemcpyDeviceToHost));
		// Check the neuron spikes array
		for (int n=0; n < test_neurons.total_number_of_neurons; n++){
			for (int i=0; i < ((max_delay / 8)+ 1); i ++){
				for (int j=0; j < 8; j++){
					int check = (neuron_spike_array[n*((max_delay / 8)+ 1) + i] >> j) & 1;
					if ((n == indices[0] || n == indices[1] || n == indices[2])
						 && ((j + i*8) == 9)){
						REQUIRE(check == 1);
					} else {
						REQUIRE(check == 0);
					}
				}
			}
		}
	}
}



/**
		INPUTSPIKINGNEURONS Test Set
**/
#include "Spike/Neurons/InputSpikingNeurons.hpp"
// No tests required yet. This class is almost entirely empty.



/**
		GENERATORINPUTSPIKINGNEURONS Test Set
**/
#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
TEST_CASE("Generator Input Spiking Neurons Class") {
	// Create an instance of the neuron class
	GeneratorInputSpikingNeurons test_neurons;

	input_spiking_neuron_parameters_struct params;
	int dim1 = 1;
	int dim2 = 10;
	float resting_pot = -50.0f;
	float threshold = 50.0f;

	params.group_shape[0] = dim1;
	params.group_shape[1] = dim2;
	params.resting_potential_v0 = resting_pot;
	params.threshold_for_action_potential_spike = threshold;

	int ID = test_neurons.AddGroup(&params);

	SECTION("Generator AddGroup Check") {
		REQUIRE(ID < 0);
	}

	// Creating a set of spikes for our group
	int num_spikes = 5;
	int neuron_ids[5] = {0, 1, 3, 6, 7};
	float spike_times[5] = {0.1f, 0.3f, 0.2f, 0.5f, 0.9f};

	test_neurons.AddStimulus(num_spikes, neuron_ids, spike_times);

	SECTION("Generator AddStimulus") {
		REQUIRE(test_neurons.total_number_of_input_stimuli == 1);
		REQUIRE(test_neurons.length_of_longest_stimulus == 5);
		REQUIRE(test_neurons.number_of_spikes_in_stimuli[0] == 5);
		for (int i=0; i < num_spikes; i++){
			REQUIRE(test_neurons.neuron_id_matrix_for_stimuli[0][i] == neuron_ids[i]);
			REQUIRE(test_neurons.spike_times_matrix_for_stimuli[0][i] == spike_times[i]);
		}
	}

	SECTION("Low Fidelity check_for_neuron_spikes Kernel Check") {
		InputSpikingNeurons* cast_test_neurons = &test_neurons;
		cast_test_neurons->allocate_device_pointers(0, false);
		cast_test_neurons->set_threads_per_block_and_blocks_per_grid(512);

		for (int s=0; s < num_spikes; s++){		
			float current_time = spike_times[s];
			float timestep = 0.1f;
			cast_test_neurons->copy_constants_to_device();
			cast_test_neurons->reset_neuron_activities();
			cast_test_neurons->check_for_neuron_spikes(current_time, timestep);

			// Copying the data to Host
			float* last_neuron_spike_times;
			last_neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(last_neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));

			for (int i=0; i < dim1*dim2; i++){
				if (i == neuron_ids[s]){
					REQUIRE(last_neuron_spike_times[i] == spike_times[s]);
				} else {
					REQUIRE(last_neuron_spike_times[i] == -1000.0f);
				}
			}
		}
	}

	SECTION("High Fidelity check_for_neuron_spikes Kernel Check") {
		int max_delay = 10;
		InputSpikingNeurons* cast_test_neurons = &test_neurons;
		cast_test_neurons->allocate_device_pointers(max_delay, true);
		cast_test_neurons->set_threads_per_block_and_blocks_per_grid(512);

		for (int s=0; s < num_spikes; s++){		
			float current_time = spike_times[s];
			float timestep = 0.1f;
			cast_test_neurons->reset_neuron_activities();
			cast_test_neurons->check_for_neuron_spikes(current_time, timestep);

			// Copy back the correct array
			char* neuron_spike_array;
			neuron_spike_array = (char*)malloc(sizeof(char)*test_neurons.total_number_of_neurons*((max_delay / 8)+ 1));
			CudaSafeCall(cudaMemcpy(neuron_spike_array, test_neurons.d_bitarray_of_neuron_spikes, sizeof(char)*test_neurons.total_number_of_neurons*((max_delay / 8)+ 1), cudaMemcpyDeviceToHost));

			for (int n=0; n < test_neurons.total_number_of_neurons; n++){
				for (int i=0; i < ((max_delay / 8)); i++){
					for (int j=0; j < 8; j++){
						int check = (neuron_spike_array[n*((max_delay / 8)+ 1) + i] >> j) & 1;
						if ((n == neuron_ids[s]) && ((j + i*8) == (int)(spike_times[s]*10.0f))){
							REQUIRE(check == 1);
						}
						else {
							REQUIRE(check == 0);
						}
					}
				}
			}
		}
	}
}



/**
		IZHIKEVICHSPIKINGNEURONS Test Set
**/
#include "Spike/Neurons/IzhikevichSpikingNeurons.hpp"
TEST_CASE("Izhikevich Spiking Neurons Class") {

	// Create an instance of the neuron class
	IzhikevichSpikingNeurons test_neurons;

	izhikevich_spiking_neuron_parameters_struct params;
	int dim1 = 1;
	int dim2 = 10;
	float resting_pot = -55.0f;
	float threshold = 50.0f;

	params.group_shape[0] = dim1;
	params.group_shape[1] = dim2;
	params.resting_potential_v0 = resting_pot;
	params.threshold_for_action_potential_spike = threshold;

	// Testing values
	params.parama = 0.02f;
	params.paramb = -0.01;
	params.paramd = 6.0f;

	// AddGroup
	int ID = test_neurons.AddGroup(&params);

	SECTION("AddGroup Testing"){
		REQUIRE(ID == 0);
		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
			REQUIRE(test_neurons.param_a[i] == 0.02f);
			REQUIRE(test_neurons.param_b[i] == -0.01f);
			REQUIRE(test_neurons.param_d[i] == 6.0f);
		}
	}

	SECTION("State Updates"){
		SpikingNeurons* cast_test_neurons = &test_neurons;
		cast_test_neurons->allocate_device_pointers(0, false);
		cast_test_neurons->set_threads_per_block_and_blocks_per_grid(512);
		cast_test_neurons->copy_constants_to_device();
		cast_test_neurons->reset_neuron_activities();

		SECTION("Initial State Variable Values"){
			// Testing before any spikes
			// Copying the data to Host
			float* state_u;
			state_u = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(state_u, test_neurons.d_states_u, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				REQUIRE(state_u[i] == 0.0f);
			}

			float* state_v;
			state_v = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(state_v, test_neurons.d_membrane_potentials_v, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				REQUIRE(state_v[i] == resting_pot);
			}
		}

		// Setting some neurons as fired
		float current_time = 0.1f;
		float timestep = 0.1f;
		int neurons[5] = {0, 3, 5, 6, 9};
		float* last_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		float* current_injection = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
			last_spike_times[i] = 0.0f;
			current_injection[i] = 0.0f;
		}
		for (int i=0; i < 5; i++){
			last_spike_times[neurons[i]] = current_time;
			current_injection[neurons[i]] = 10.0f;
		}
		// Copy to device
		CudaSafeCall(cudaMemcpy(test_neurons.d_last_spike_time_of_each_neuron, last_spike_times, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(test_neurons.d_current_injections, current_injection, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyHostToDevice));
		
		// Running update of state u
		cast_test_neurons->check_for_neuron_spikes(current_time, timestep);
	
		SECTION("Checking State U spike reset"){
			float* state_u;
			state_u = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			// Check the resulting state values
			CudaSafeCall(cudaMemcpy(state_u, test_neurons.d_states_u, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				if ((i == neurons[0]) || (i == neurons[1]) || (i == neurons[2]) || (i == neurons[3]) || (i == neurons[4])){
					REQUIRE(state_u[i] == params.paramd);
				} else {
					REQUIRE(state_u[i] == 0.0f);
				}
			}
		}

		cast_test_neurons->update_membrane_potentials(timestep, current_time);

		SECTION("Membrane Potential Update"){
			// Computing what I expect the values to be:
			float* u = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			float* v = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			float* v_update = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				u[i] = 0.0f;
				v[i] = resting_pot;
				v_update[i] = 0.0f;
			}
			// Updating u as before
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				if ((i == neurons[0]) || (i == neurons[1]) || (i == neurons[2]) || (i == neurons[3]) || (i == neurons[4])){
					u[i] += params.paramd;
				} else {
					u[i] = 0.0f;
				}
			}
			// Computing v_update, v and u:
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				v_update[i] = 0.04f*v[i]*v[i] + 5.0f*v[i] + 140.0f - u[i] + current_injection[i];
				v[i] += 1000.0f*timestep*v_update[i];
				u[i] += 1000.0f*timestep*(params.parama*(params.paramb*v[i] - u[i]));
			}

			// Copying the data from device
			float* state_u = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(state_u, test_neurons.d_states_u, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));

			float* state_v = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
			CudaSafeCall(cudaMemcpy(state_v, test_neurons.d_membrane_potentials_v, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));

			// Compare results
			for (int i=0; i < test_neurons.total_number_of_neurons; i++){
				REQUIRE(state_u[i] == u[i]);
				REQUIRE(state_v[i] == v[i]);
			}
		}	
	}
}


/**
		POISSONINPUTSPIKINGNEURONS Test Set
**/
#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
TEST_CASE("Poisson Input Spiking Neurons Class") {

	// Create an instance of the neuron class
	PoissonInputSpikingNeurons test_neurons;
        test_neurons.setup_random_states_on_device();

	poisson_input_spiking_neuron_parameters_struct params;
	int dim1 = 1;
	int dim2 = 300;
	float resting_pot = -55.0f;
	float threshold = 50.0f;

	params.group_shape[0] = dim1;
	params.group_shape[1] = dim2;
	params.resting_potential_v0 = resting_pot;
	params.threshold_for_action_potential_spike = threshold;

	params.rate = 30.0f;

	float timestep = 0.01;

	// AddGroup
	int ID = test_neurons.AddGroup(&params);

	////////// SET UP STATES FOR RANDOM STATE MANAGER SINGLETON ///////////
        /*
	int random_states_threads_per_block_x = 128;
	int random_states_number_of_blocks_x = 64;
	RandomStateManager::instance()->set_up_random_states(random_states_threads_per_block_x, random_states_number_of_blocks_x, 9);
        */
        test_neurons.set_up_rates();


	SECTION("AddGroup"){
		REQUIRE(ID < 0);
		// Checking that the rate has been correctly set
		for(int i=0; i < test_neurons.total_number_of_neurons; i++){
			REQUIRE(test_neurons.rates[i] == params.rate);
		}
	}


	SECTION("Membrane Potential Update"){
		InputSpikingNeurons* cast_test_neurons = &test_neurons;
		cast_test_neurons->allocate_device_pointers(0, false);
		cast_test_neurons->set_threads_per_block_and_blocks_per_grid(512);
		cast_test_neurons->copy_constants_to_device();
		cast_test_neurons->reset_neuron_activities();

		float current_time = 0.9f;

		// Running the network for many timesteps and checking rate
		cast_test_neurons->update_membrane_potentials(timestep, current_time);

		// Copying the membrane potentials back
		float* membrane_potentials;
		membrane_potentials = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(membrane_potentials, test_neurons.d_membrane_potentials_v, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Count the number of spikes
		int count = 0;
		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
			if (membrane_potentials[i] > threshold){
				count++;
			}
		}
		// Checking that the rate for the combined population is within 5%
		float actual_rate = count / (test_neurons.total_number_of_neurons * timestep);
		REQUIRE(std::abs(actual_rate - params.rate) < 5.0f);
	}
}


/**
		LIFSPIKINGNEURONS Test Set
**/
// TO DO
#include "Spike/Neurons/LIFSpikingNeurons.hpp"
TEST_CASE("LIF Spiking Neurons Class") {
}

/**
		IMAGEPOISSONINPUTSPIKINGNEURONS Test Set
**/
// TO DO
#include "Spike/Neurons/ImagePoissonInputSpikingNeurons.hpp"
TEST_CASE("Image Poisson Input Spiking Neurons Class") {
}

/**
		ADEXSPIKINGNEURONS Test Set
**/
// TO DO
#include "Spike/Neurons/AdExSpikingNeurons.hpp"
TEST_CASE("Adaptive Exponential Spiking Neurons Class") {
}

