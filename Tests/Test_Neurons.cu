#include "../catch.hpp"
#include "../Helpers/CUDAErrorCheckHelpers.h"



/**
		NEURONS.CU Test Set
**/
#include "../Neurons/Neurons.h"
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
		SPIKINGNEURONS.CU Test Set
**/
#include "../Neurons/SpikingNeurons.h"
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
		test_neurons.allocate_device_pointers(0, false);
		// Selecting some indices to set to fire
		int indices[3];
		indices[0] = 1; indices[1] = 3; indices[2] = 5;
		for (int i=0; i < 3; i++){
			test_neurons.after_spike_reset_membrane_potentials_c[indices[i]] = 60.0f;
		}
		// Copying the given arrays to the GPU
		test_neurons.reset_neurons();
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
		test_neurons.allocate_device_pointers(max_delay, true);
		// Selecting some indices to set to fire
		int indices[3];
		indices[0] = 1; indices[1] = 3; indices[2] = 5;
		for (int i=0; i < 3; i++){
			test_neurons.after_spike_reset_membrane_potentials_c[indices[i]] = 60.0f;
		}
		// Copying the given arrays to the GPU
		test_neurons.reset_neurons();
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
		INPUTSPIKINGNEURONS.CU Test Set
**/

// No tests required yet. This class if almost entirely empty.



/**
		GENERATORINPUTSPIKINGNEURONS.CU Test Set
**/
#include "../Neurons/GeneratorInputSpikingNeurons.h"
TEST_CASE("Generator Input Spiking Neurons Class") {
	// Create an instance of the neuron class
	GeneratorInputSpikingNeurons test_neurons;

	spiking_neuron_parameters_struct params;
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
		test_neurons.allocate_device_pointers(0, false);

		for (int s=0; s < num_spikes; s++){		
			float current_time = spike_times[s];
			float timestep = 0.1f;
			test_neurons.reset_neurons();
			test_neurons.check_for_neuron_spikes(current_time, timestep);

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
		test_neurons.allocate_device_pointers(max_delay, true);

		for (int s=0; s < num_spikes; s++){		
			float current_time = spike_times[s];
			float timestep = 0.1f;
			test_neurons.reset_neurons();
			test_neurons.check_for_neuron_spikes(current_time, timestep);

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




